use crate::config::OllamaConfig;
use crate::error::{Result, SoasError};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tracing::{debug, info, warn};

/// Trunca un string de forma segura respetando límites de caracteres UTF-8
fn safe_truncate(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn normalize_filename_title(filename: &str) -> String {
    filename
        .replace(['_', '-', '.'], " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn derive_keywords_simple(text: &str, max_keywords: usize) -> Vec<String> {
    let stopwords = [
        "de", "del", "la", "las", "el", "los", "y", "o", "en", "con", "para", "por", "un",
        "una", "que", "es", "al", "se", "sin", "archivo", "documento",
    ];

    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for token in text
        .split(|c: char| !c.is_alphanumeric())
        .map(|w| w.trim().to_lowercase())
        .filter(|w| w.len() >= 3)
    {
        if stopwords.contains(&token.as_str()) {
            continue;
        }
        if seen.insert(token.clone()) {
            out.push(token);
        }
        if out.len() >= max_keywords {
            break;
        }
    }

    out
}

fn extract_json_object(raw: &str) -> Option<&str> {
    let start = raw.find('{')?;
    let end = raw.rfind('}')?;
    if end <= start {
        return None;
    }
    Some(&raw[start..=end])
}

/// Cliente para interactuar con Ollama (embeddings, chat, visión)
pub struct OllamaClient {
    client: Client,
    config: OllamaConfig,
    /// Circuit breaker: se activa cuando vision_model (glm-ocr) falla con error de red.
    /// Evita perder ~5 min de timeout por cada imagen restante.
    vision_model_failed: AtomicBool,
    /// Circuit breaker para description_model (qwen3-vl).
    /// Si falla 2 veces consecutivas con error de red/timeout, se desactiva
    /// para no bloquear el pipeline con imágenes que jamás se procesarán.
    description_model_failed: AtomicBool,
    /// Contador de fallos consecutivos de description_model.
    description_fail_count: std::sync::atomic::AtomicU32,
    /// Flag para saltar OCR y usar solo description_model (moondream).
    /// Se activa durante `reimages` para ir directo a descripción visual.
    skip_ocr: AtomicBool,
}

// ─────────────────────────────────────────────
//  Request / Response DTOs
// ─────────────────────────────────────────────

#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<String>,
    /// Opciones de runtime para Ollama (num_ctx, temperature, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<serde_json::Value>,
    /// Tiempo que Ollama mantiene el modelo en RAM después del request.
    /// "10m" = 10 minutos (default de Ollama es 5m).
    /// Mantenerlo alto evita recargar el modelo entre requests consecutivos.
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<String>,
}

#[derive(Serialize, Deserialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct ChatResponse {
    message: ChatResponseMessage,
}

#[derive(Deserialize)]
struct ChatResponseMessage {
    content: String,
}

impl OllamaClient {
    fn chat_runtime_options() -> serde_json::Value {
        serde_json::json!({
            "num_ctx": 3072,
            "num_predict": 140,
            "temperature": 0.2
        })
    }

    fn chat_json_runtime_options() -> serde_json::Value {
        serde_json::json!({
            "num_ctx": 3072,
            "num_predict": 320,
            "temperature": 0.1
        })
    }

    /// Opciones ultra-ligeras para búsqueda interactiva.
    /// La query del usuario es corta (~10-30 tokens) y la respuesta JSON es mínima,
    /// así que num_ctx=1024 y num_predict=100 son MÁS que suficientes.
    /// temperature=0: respuestas deterministas para la misma consulta.
    fn chat_search_runtime_options() -> serde_json::Value {
        serde_json::json!({
            "num_ctx": 1024,
            "num_predict": 100,
            "temperature": 0.0
        })
    }

    fn fallback_file_description(filename: &str, content_preview: &str) -> FileDescription {
        let title = normalize_filename_title(filename);
        let excerpt = safe_truncate(content_preview.trim(), 220);
        let description = if excerpt.is_empty() {
            format!("Archivo {}", title)
        } else {
            format!("{}", excerpt)
        };

        let mut keyword_source = title.clone();
        if !excerpt.is_empty() {
            keyword_source.push(' ');
            keyword_source.push_str(excerpt);
        }

        let keywords = derive_keywords_simple(&keyword_source, 6);
        let semantic_tags = if keywords.is_empty() {
            vec!["documento".to_string()]
        } else {
            keywords.iter().take(3).cloned().collect()
        };

        FileDescription {
            title,
            description,
            keywords,
            semantic_tags,
            language: Some("es".to_string()),
            content_type_group: None,
            suggested_name: None,
            suggested_category: None,
        }
    }

    /// Crea un nuevo cliente de Ollama
    pub fn new(config: OllamaConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            config,
            vision_model_failed: AtomicBool::new(false),
            description_model_failed: AtomicBool::new(false),
            description_fail_count: std::sync::atomic::AtomicU32::new(0),
            skip_ocr: AtomicBool::new(false),
        }
    }

    /// Verifica que Ollama esté disponible
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/api/tags", self.config.base_url);
        match self.client.get(&url).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    /// Lista los modelos disponibles en Ollama
    pub async fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/api/tags", self.config.base_url);
        let resp = self.client.get(&url).send().await?;

        if !resp.status().is_success() {
            return Err(SoasError::Ollama(format!(
                "Error al listar modelos: {}",
                resp.status()
            )));
        }

        #[derive(Deserialize)]
        struct TagsResponse {
            models: Vec<ModelInfo>,
        }

        #[derive(Deserialize)]
        struct ModelInfo {
            name: String,
        }

        let tags: TagsResponse = resp.json().await?;
        Ok(tags.models.into_iter().map(|m| m.name).collect())
    }

    // ─────────────────────────────────────────
    //  Embeddings
    // ─────────────────────────────────────────

    /// Genera embeddings para uno o más textos
    pub async fn generate_embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let url = format!("{}/api/embed", self.config.base_url);

        let request = EmbeddingRequest {
            model: self.config.embedding_model.clone(),
            input: texts.to_vec(),
        };

        debug!(
            "Generando embeddings para {} textos con modelo {}",
            texts.len(),
            self.config.embedding_model
        );

        let resp = self.client.post(&url).json(&request).send().await?;

        if !resp.status().is_success() {
            let error_text = resp.text().await.unwrap_or_default();
            return Err(SoasError::Ollama(format!(
                "Error al generar embeddings: {}",
                error_text
            )));
        }

        let embedding_resp: EmbeddingResponse = resp.json().await?;

        debug!(
            "Embeddings generados: {} vectores de dimensión {}",
            embedding_resp.embeddings.len(),
            embedding_resp
                .embeddings
                .first()
                .map(|v| v.len())
                .unwrap_or(0)
        );

        Ok(embedding_resp.embeddings)
    }

    /// Genera un embedding para un solo texto
    pub async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.generate_embeddings(&[text.to_string()]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| SoasError::Embedding("No se generó ningún embedding".to_string()))
    }

    // ─────────────────────────────────────────
    //  Chat / LLM
    // ─────────────────────────────────────────

    /// Envía un prompt al modelo de chat y devuelve la respuesta
    pub async fn chat(&self, system_prompt: &str, user_message: &str) -> Result<String> {
        let url = format!("{}/api/chat", self.config.base_url);

        let request = ChatRequest {
            model: self.config.chat_model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system_prompt.to_string(),
                    images: None,
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: user_message.to_string(),
                    images: None,
                },
            ],
            stream: false,
            format: None,
            options: Some(Self::chat_runtime_options()),
            keep_alive: None,
        };

        let resp = self.client.post(&url).json(&request).send().await?;

        if !resp.status().is_success() {
            let error_text = resp.text().await.unwrap_or_default();
            return Err(SoasError::Ollama(format!("Error de chat: {}", error_text)));
        }

        let chat_resp: ChatResponse = resp.json().await?;
        Ok(chat_resp.message.content)
    }

    /// Envía un prompt con formato JSON esperado
    pub async fn chat_json(&self, system_prompt: &str, user_message: &str) -> Result<String> {
        let url = format!("{}/api/chat", self.config.base_url);

        let request = ChatRequest {
            model: self.config.chat_model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system_prompt.to_string(),
                    images: None,
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: user_message.to_string(),
                    images: None,
                },
            ],
            stream: false,
            format: Some("json".to_string()),
            options: Some(Self::chat_json_runtime_options()),
            keep_alive: None,
        };

        let resp = self.client.post(&url).json(&request).send().await?;

        if !resp.status().is_success() {
            let error_text = resp.text().await.unwrap_or_default();
            return Err(SoasError::Ollama(format!(
                "Error de chat JSON: {}",
                error_text
            )));
        }

        let chat_resp: ChatResponse = resp.json().await?;
        Ok(chat_resp.message.content)
    }

    // ─────────────────────────────────────────
    //  Control de visión (circuit breaker, skip OCR)
    // ─────────────────────────────────────────

    /// Activa/desactiva el modo "saltar OCR": va directo a description_model.
    /// Se usa durante `reimages` para evitar el timeout de glm-ocr.
    pub fn set_skip_ocr(&self, skip: bool) {
        self.skip_ocr.store(skip, Ordering::Relaxed);
    }

    /// ¿Se debe saltar el paso de OCR (glm-ocr)?
    /// True si: se activó `skip_ocr` (reimages) o el circuit breaker se disparó.
    pub fn should_skip_ocr(&self) -> bool {
        self.skip_ocr.load(Ordering::Relaxed)
            || self.vision_model_failed.load(Ordering::Relaxed)
    }

    /// Resetea los circuit breakers de los modelos de visión.
    pub fn reset_vision_breaker(&self) {
        self.vision_model_failed.store(false, Ordering::Relaxed);
        self.description_model_failed.store(false, Ordering::Relaxed);
        self.description_fail_count.store(0, Ordering::Relaxed);
    }

    // ─────────────────────────────────────────
    //  Visión (OCR)
    // ─────────────────────────────────────────

    /// Analiza una imagen usando el modelo de visión configurado (`vision_model`, por defecto glm-ocr).
    /// Incluye circuit breaker: si falla con error de red, se desactiva para no bloquear
    /// ~5 min de timeout en cada imagen restante.
    pub async fn analyze_image(&self, image_base64: &str, prompt: &str) -> Result<String> {
        // Circuit breaker: si ya falló, no intentar (ahorra ~5 min de timeout)
        if self.vision_model_failed.load(Ordering::Relaxed) {
            return Err(SoasError::Ollama(format!(
                "Circuit breaker activo: {} no disponible",
                self.config.vision_model
            )));
        }

        match self
            .analyze_image_with_model(&self.config.vision_model.clone(), image_base64, prompt)
            .await
        {
            Ok(result) => Ok(result),
            Err(e) => {
                // Si es error de red, activar circuit breaker
                if matches!(&e, SoasError::Network(_)) {
                    warn!(
                        "Circuit breaker activado: {} no responde. \
                         Se usará {} para las imágenes restantes.",
                        self.config.vision_model, self.config.description_model
                    );
                    self.vision_model_failed.store(true, Ordering::Relaxed);
                }
                Err(e)
            }
        }
    }

    /// Describe una foto/ilustración usando el modelo de descripción (`description_model`, por defecto qwen3-vl:2b).
    /// Se usa como fallback cuando glm-ocr no encuentra texto suficiente.
    /// Incluye circuit breaker: si falla 2 veces consecutivas, se desactiva.
    pub async fn describe_photo(&self, image_base64: &str, prompt: &str) -> Result<String> {
        // Circuit breaker: si ya falló 2+ veces, no intentar
        if self.description_model_failed.load(Ordering::Relaxed) {
            return Err(SoasError::Ollama(format!(
                "Circuit breaker activo: {} no disponible (falló {} veces)",
                self.config.description_model,
                self.description_fail_count.load(Ordering::Relaxed)
            )));
        }

        match self
            .analyze_image_with_model(&self.config.description_model.clone(), image_base64, prompt)
            .await
        {
            Ok(result) => {
                // Solo resetear si la respuesta tiene contenido útil.
                // Respuestas vacías son "soft failures": el modelo respondió
                // pero no generó nada → síntoma de que no puede procesar.
                if !result.trim().is_empty() {
                    self.description_fail_count.store(0, Ordering::Relaxed);
                } else {
                    let count = self.description_fail_count.fetch_add(1, Ordering::Relaxed) + 1;
                    warn!(
                        "Respuesta vacía de {} ({} veces consecutivas)",
                        self.config.description_model, count
                    );
                    if count >= 3 {
                        warn!(
                            "Circuit breaker activado: {} devuelve vacío consistentemente. \
                             Se usará fallback para imágenes restantes.",
                            self.config.description_model
                        );
                        self.description_model_failed.store(true, Ordering::Relaxed);
                    }
                }
                Ok(result)
            }
            Err(e) => {
                // Si es error de red/timeout, incrementar contador
                if matches!(&e, SoasError::Network(_)) {
                    let count = self.description_fail_count.fetch_add(1, Ordering::Relaxed) + 1;
                    if count >= 2 {
                        warn!(
                            "Circuit breaker activado: {} falló {} veces consecutivas. \
                             Se saltarán las imágenes restantes.",
                            self.config.description_model, count
                        );
                        self.description_model_failed.store(true, Ordering::Relaxed);
                    }
                }
                Err(e)
            }
        }
    }

    /// Analiza una imagen usando un modelo de visión explícito.
    /// Usa `vision_timeout_secs` (más corto que el global) para no bloquear si el modelo falla.
    pub async fn analyze_image_with_model(
        &self,
        model: &str,
        image_base64: &str,
        prompt: &str,
    ) -> Result<String> {
        let url = format!("{}/api/chat", self.config.base_url);

        let request = ChatRequest {
            model: model.to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
                images: Some(vec![image_base64.to_string()]),
            }],
            stream: false,
            format: None,
            // num_ctx=2048: suficiente para OCR/descripción de una imagen.
            // Valores altos (16384) desperdician RAM y hacen la inferencia mucho
            // más lenta en CPU. El input de imagen es fijo (~600 tokens para 768px)
            // y la respuesta rara vez supera 300 tokens.
            // num_predict=256: limita la salida para que no genere texto infinito.
            options: Some(serde_json::json!({
                "num_ctx": 2048,
                "num_predict": 256
            })),
            // Mantener modelo en RAM 10 min: las imágenes se procesan agrupadas,
            // así el modelo se carga UNA vez y se reutiliza para todo el lote.
            keep_alive: Some("10m".to_string()),
        };

        // Timeout específico para visión (más corto que el global de 300s)
        let resp = self
            .client
            .post(&url)
            .json(&request)
            .timeout(Duration::from_secs(self.config.vision_timeout_secs))
            .send()
            .await?;

        if !resp.status().is_success() {
            let error_text = resp.text().await.unwrap_or_default();
            return Err(SoasError::Ollama(format!(
                "Error de visión [{}]: {}",
                model, error_text
            )));
        }

        let chat_resp: ChatResponse = resp.json().await?;
        Ok(chat_resp.message.content)
    }

    // ─────────────────────────────────────────
    //  Funciones de alto nivel para SOAS
    // ─────────────────────────────────────────

    /// Genera una descripción y keywords para un archivo dado su contenido.
    /// Con `use_llm_enrichment` activo, esto enriquece los embeddings.
    pub async fn describe_file(
        &self,
        filename: &str,
        content_preview: &str,
    ) -> Result<FileDescription> {
        let system_prompt = r#"Genera SOLO JSON válido en español con:
    - "title": título corto
    - "description": resumen semántico (1 oración)
    - "keywords": 4-8 palabras clave
    - "semantic_tags": 2-4 temas
    - "language": "es" | "en" | etc.
    - "content_type_group": "documento" | "imagen" | "hoja_calculo" | "codigo" | "presentacion" | "comprimido" | "archivo"
    - "suggested_name": string opcional
    - "suggested_category": string opcional

    No incluyas texto fuera del JSON."#;

        let user_message = format!(
            "Archivo: {}\n\nContenido (primeros caracteres):\n{}",
            filename,
            safe_truncate(content_preview, 500)
        );

        let response = self.chat_json(system_prompt, &user_message).await?;

        let mut desc: FileDescription = match serde_json::from_str(&response) {
            Ok(parsed) => parsed,
            Err(first_err) => {
                if let Some(json_block) = extract_json_object(&response) {
                    match serde_json::from_str::<FileDescription>(json_block) {
                        Ok(parsed) => parsed,
                        Err(second_err) => {
                            warn!(
                                "Error parseando descripción de archivo (directo: {}, extraído: {}). Usando fallback local.",
                                first_err,
                                second_err
                            );
                            debug!("Respuesta LLM raw: {}", safe_truncate(&response, 500));
                            Self::fallback_file_description(filename, content_preview)
                        }
                    }
                } else {
                    warn!(
                        "Respuesta describe_file sin JSON utilizable ({}). Usando fallback local.",
                        first_err
                    );
                    debug!("Respuesta LLM raw: {}", safe_truncate(&response, 500));
                    Self::fallback_file_description(filename, content_preview)
                }
            }
        };

        if desc.title.trim().is_empty() {
            desc.title = normalize_filename_title(filename);
        }
        if desc.description.trim().is_empty() {
            desc.description = safe_truncate(content_preview, 220).to_string();
        }
        if desc.keywords.is_empty() {
            let mut source = desc.title.clone();
            source.push(' ');
            source.push_str(&desc.description);
            desc.keywords = derive_keywords_simple(&source, 6);
        }
        if desc.semantic_tags.is_empty() {
            desc.semantic_tags = desc.keywords.iter().take(3).cloned().collect();
        }

        info!(
            "LLM describe_file: title={:?}, keywords={:?}, tags={:?}",
            desc.title, desc.keywords, desc.semantic_tags
        );

        Ok(desc)
    }

    /// Envía un prompt JSON con opciones de búsqueda (ultra-ligeras).
    /// Diseñado para queries interactivas donde la latencia es crítica.
    async fn chat_json_search(&self, system_prompt: &str, user_message: &str) -> Result<String> {
        let url = format!("{}/api/chat", self.config.base_url);

        let request = ChatRequest {
            model: self.config.chat_model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system_prompt.to_string(),
                    images: None,
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: user_message.to_string(),
                    images: None,
                },
            ],
            stream: false,
            format: Some("json".to_string()),
            options: Some(Self::chat_search_runtime_options()),
            keep_alive: None,
        };

        let resp = self.client.post(&url).json(&request).send().await?;

        if !resp.status().is_success() {
            let error_text = resp.text().await.unwrap_or_default();
            return Err(SoasError::Ollama(format!(
                "Error de chat JSON search: {}",
                error_text
            )));
        }

        let chat_resp: ChatResponse = resp.json().await?;
        Ok(chat_resp.message.content)
    }

    /// Mejora una consulta de búsqueda extrayendo keywords y tipo de archivo.
    ///
    /// Prompt ultra-compacto: ~150 tokens de input vs ~600 del anterior.
    /// Genera ~50-80 tokens de output. En CPU con num_ctx=1024, tarda ~3-5s vs ~16s.
    pub async fn enhance_search_query(&self, user_query: &str) -> Result<EnhancedQuery> {
        let system_prompt = r#"Extrae de la búsqueda de archivos un JSON con:
- "keywords": 2-5 palabras clave para buscar en texto de archivos
- "file_types": extensiones probables (ej: ["pdf"], ["jpg","png"], ["docx"]); vacío si no es claro
- "hard_type_filter": true solo si pide explícitamente un tipo ("fotos", "PDFs")
Solo JSON."#;

        let response = self.chat_json_search(system_prompt, user_query).await?;

        // Parsear respuesta mínima y construir EnhancedQuery
        #[derive(Deserialize)]
        struct MinimalQuery {
            #[serde(default, deserialize_with = "deserialize_string_or_vec")]
            keywords: Vec<String>,
            #[serde(default, deserialize_with = "deserialize_string_or_vec")]
            file_types: Vec<String>,
            #[serde(default)]
            hard_type_filter: bool,
        }

        let minimal: MinimalQuery = serde_json::from_str(&response).map_err(|e| {
            warn!("Error parseando consulta mejorada: {}", e);
            SoasError::Ollama(format!(
                "Respuesta no es JSON válido: {}",
                safe_truncate(&response, 200)
            ))
        })?;

        Ok(EnhancedQuery {
            reasoning: String::new(),
            enhanced_query: String::new(), // ya no se usa para embedding
            keywords: minimal.keywords,
            file_types: minimal.file_types,
            content_type: String::new(),
            date_hints: Vec::new(),
            hard_type_filter: minimal.hard_type_filter,
        })
    }

    /// Sugiere categorías para un archivo basado en su contenido
    pub async fn suggest_categories(
        &self,
        filename: &str,
        content_preview: &str,
        existing_categories: &[String],
    ) -> Result<Vec<String>> {
        let categories_list = if existing_categories.is_empty() {
            "No hay categorías existentes.".to_string()
        } else {
            format!(
                "Categorías existentes: {}",
                existing_categories.join(", ")
            )
        };

        let system_prompt = format!(
            r#"Eres un asistente de organización de archivos.
Dado un archivo, sugiere en qué categorías debería clasificarse.
{}
Si ninguna categoría existente aplica, sugiere nuevas categorías.

Responde con un JSON: {{"categories": ["cat1", "cat2"]}}"#,
            categories_list
        );

        let user_message = format!(
            "Archivo: {}\nContenido: {}",
            filename,
            safe_truncate(content_preview, 1000)
        );

        let response = self
            .chat_json(&system_prompt, &user_message)
            .await?;

        #[derive(Deserialize)]
        struct CategoriesResponse {
            categories: Vec<String>,
        }

        let cats: CategoriesResponse = serde_json::from_str(&response).map_err(|e| {
            SoasError::Ollama(format!("Error parseando categorías: {}", e))
        })?;

        Ok(cats.categories)
    }

    /// Devuelve la dimensión configurada de embedding
    pub fn embedding_dimensions(&self) -> usize {
        self.config.embedding_dimensions
    }
}

/// Descripción generada por LLM para un archivo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileDescription {
    pub title: String,
    pub description: String,
    /// Keywords en español para búsqueda.
    /// qwen2.5:3b a veces devuelve un string en vez de array, se maneja con deserialize_string_or_vec.
    #[serde(default, deserialize_with = "deserialize_string_or_vec")]
    pub keywords: Vec<String>,
    /// Etiquetas semánticas amplias (temas, conceptos, entidades)
    #[serde(default, deserialize_with = "deserialize_string_or_vec")]
    pub semantic_tags: Vec<String>,
    #[serde(default)]
    pub language: Option<String>,
    /// Grupo de tipo de contenido detectado
    #[serde(default)]
    pub content_type_group: Option<String>,
    #[serde(default)]
    pub suggested_name: Option<String>,
    #[serde(default)]
    pub suggested_category: Option<String>,
}

/// Consulta de búsqueda mejorada por LLM con cadena de pensamiento
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedQuery {
    /// Razonamiento CoT del LLM sobre la intención del usuario
    #[serde(default)]
    pub reasoning: String,
    /// Consulta reformulada para mejor matching semántico
    pub enhanced_query: String,
    /// Keywords extraídas
    #[serde(default, deserialize_with = "deserialize_string_or_vec")]
    pub keywords: Vec<String>,
    /// Extensiones de archivo probables
    #[serde(default, deserialize_with = "deserialize_string_or_vec")]
    pub file_types: Vec<String>,
    /// Tipo de contenido detectado
    #[serde(default)]
    pub content_type: String,
    /// Pistas temporales detectadas
    #[serde(default, deserialize_with = "deserialize_string_or_vec")]
    pub date_hints: Vec<String>,
    /// True si el usuario claramente busca un tipo específico de archivo
    #[serde(default)]
    pub hard_type_filter: bool,
}

/// Deserializa un campo que puede venir como string vacío "" o como array []
fn deserialize_string_or_vec<'de, D>(deserializer: D) -> std::result::Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;

    struct StringOrVec;

    impl<'de> de::Visitor<'de> for StringOrVec {
        type Value = Vec<String>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a string or array of strings")
        }

        fn visit_str<E: de::Error>(self, value: &str) -> std::result::Result<Vec<String>, E> {
            if value.is_empty() {
                Ok(Vec::new())
            } else if value.contains(',') {
                // qwen2.5:3b a veces devuelve "word1, word2, word3" en vez de ["word1","word2","word3"]
                Ok(value.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
            } else {
                Ok(vec![value.to_string()])
            }
        }

        fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<Vec<String>, A::Error> {
            let mut vec = Vec::new();
            while let Some(val) = seq.next_element::<String>()? {
                vec.push(val);
            }
            Ok(vec)
        }
    }

    deserializer.deserialize_any(StringOrVec)
}
