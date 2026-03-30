use crate::config::OllamaConfig;
use crate::error::{Result, SoasError};
use reqwest::Client;
use serde::{Deserialize, Serialize};
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
        "una", "que", "es", "al", "se", "sin", "archivo", "documento", "the", "and", "for",
        "this", "that", "with", "from", "are", "was", "has", "have", "not", "but",
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

/// Cliente para interactuar con Ollama (embeddings, chat, visión).
///
/// Modelos soportados:
/// - **qwen3:1.7b**: chat y clasificación de archivos
/// - **llava-phi3**: visión (descripción de imágenes)
/// - **nomic-embed-text**: embeddings semánticos 768d
pub struct OllamaClient {
    client: Client,
    config: OllamaConfig,
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
    /// Opciones de runtime para Ollama (num_ctx, temperature, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<serde_json::Value>,
    /// Tiempo que Ollama mantiene el modelo en RAM después del request.
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<String>,
    /// Desactivar modo thinking de qwen3.
    /// Si false, el modelo no genera <think>...</think> y usa todos los
    /// tokens de num_predict para la respuesta real.
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<bool>,
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
    /// Opciones de runtime para llava-phi3 (visión).
    /// num_predict=512: suficiente para descripciones breves.
    /// temperature=0.1: respuestas consistentes y factuales.
    fn vision_runtime_options() -> serde_json::Value {
        serde_json::json!({
            "temperature": 0.1,
            "num_ctx": 3072,
            "num_predict": 512
        })
    }

    /// Opciones de runtime para qwen3:4b (chat general).
    fn chat_runtime_options() -> serde_json::Value {
        serde_json::json!({
            "num_ctx": 4096,
            "num_predict": 768,
            "temperature": 0.2
        })
    }

    /// Opciones para chat que espera respuesta JSON.
    /// num_predict=768: qwen3:4b genera JSON más rico y detallado.
    /// Con think=false (en el request), no se desperdician tokens en razonamiento.
    fn chat_json_runtime_options() -> serde_json::Value {
        serde_json::json!({
            "num_ctx": 4096,
            "num_predict": 768,
            "temperature": 0.15
        })
    }

    /// Opciones para búsqueda interactiva.
    /// temperature=0: determinista para la misma consulta.
    fn chat_search_runtime_options() -> serde_json::Value {
        serde_json::json!({
            "num_ctx": 3072,
            "num_predict": 768,
            "temperature": 0.0
        })
    }

    fn fallback_file_description(filename: &str, content_preview: &str) -> FileDescription {
        let title = normalize_filename_title(filename);
        let excerpt = safe_truncate(content_preview.trim(), 220);
        let description = if excerpt.is_empty() {
            format!("Archivo {}", title)
        } else {
            excerpt.to_string()
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
            description: description.clone(),
            summary: Some(description),
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

        Self { client, config }
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
    //  Embeddings (nomic-embed-text)
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
    //  Chat / LLM (qwen3:1.7b)
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
            options: Some(Self::chat_runtime_options()),
            keep_alive: None,
            think: None, // chat libre: thinking puede ser útil
        };

        let resp = self.client.post(&url).json(&request).send().await?;

        if !resp.status().is_success() {
            let error_text = resp.text().await.unwrap_or_default();
            return Err(SoasError::Ollama(format!("Error de chat: {}", error_text)));
        }

        let chat_resp: ChatResponse = resp.json().await?;
        Ok(chat_resp.message.content.trim().to_string())
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
            options: Some(Self::chat_json_runtime_options()),
            keep_alive: None,
            // CLAVE: desactivar thinking para JSON.
            // qwen3 por defecto genera <think>...</think> que consume
            // tokens de num_predict, dejando el JSON truncado.
            think: Some(false),
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
        Ok(chat_resp.message.content.trim().to_string())
    }

    // ─────────────────────────────────────────
    //  Visión (llava-phi3)
    // ─────────────────────────────────────────

    /// Analiza una imagen usando el modelo de visión (llava-phi3).
    /// Usa `vision_timeout_secs` para no bloquear si el modelo tarda.
    pub async fn describe_photo(&self, image_base64: &str, prompt: &str) -> Result<String> {
        let url = format!("{}/api/chat", self.config.base_url);

        let model = &self.config.vision_model;

        let request = ChatRequest {
            model: model.clone(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
                images: Some(vec![image_base64.to_string()]),
            }],
            stream: false,
            options: Some(Self::vision_runtime_options()),
            // Mantener modelo en RAM 10 min: las imágenes se procesan agrupadas.
            keep_alive: Some("10m".to_string()),
            think: None, // llava-phi3 no soporta think
        };

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
        let result = chat_resp.message.content.trim().to_string();

        if result.is_empty() {
            debug!("Respuesta de visión vacía para modelo {}", model);
        } else {
            let end = {
                let mut e = result.len().min(200);
                while e > 0 && !result.is_char_boundary(e) { e -= 1; }
                e
            };
            debug!(
                "Visión [{}]: {} chars — {:?}",
                model,
                result.len(),
                &result[..end]
            );
        }

        Ok(result)
    }

    // ─────────────────────────────────────────
    //  Funciones de alto nivel para SOAS
    // ─────────────────────────────────────────

    /// Genera una descripción semántica, keywords y etiquetas para un archivo.
    ///
    /// El prompt está diseñado para que qwen3:1.7b genere:
    /// - Keywords concretas y buscables (nombres propios, términos técnicos, acciones)
    /// - Tags semánticos que capturen el *tema* del archivo
    /// - Descripción optimizada para matching con embeddings de nomic-embed-text
    pub async fn describe_file(
        &self,
        filename: &str,
        content_preview: &str,
        folder_path: &str,
    ) -> Result<FileDescription> {
        let system_prompt = r#"Eres un experto analista de documentos especializado en indexación semántica. Tu tarea es extraer metadatos de alta calidad de archivos para un motor de búsqueda.

RESPONDE ÚNICAMENTE con un JSON válido. Sin texto antes ni después. Sin bloques ```json.

CAMPOS REQUERIDOS:
- "title": Título descriptivo y profesional (máx. 80 chars). Si el nombre de archivo es críptico, infiere un título real del contenido.
- "summary": Resumen ejecutivo de 3-5 oraciones DENSAS en información. OBLIGATORIO incluir cuando estén presentes: fechas exactas, nombres completos de personas/empresas/instituciones, montos con moneda, números de referencia/folio, propósito del documento y conclusiones.
- "keywords": Array de 10-15 términos de búsqueda de ALTA ESPECIFICIDAD. Incluye: nombres propios, términos técnicos del dominio, acrónimos, sinónimos, variantes ortográficas. EVITA palabras genéricas como "documento", "archivo", "información".
- "semantic_tags": Array de 3-6 etiquetas taxonómicas en formato "Categoría/Subcategoría" (ej: "Legal/Contrato", "Finanzas/Factura", "RRHH/Nómina", "Técnico/Manual", "Educativo/Tarea").
- "language": Código ISO 639-1 del idioma principal del contenido.
- "content_type_group": UNO de: "documento", "hoja_calculo", "presentacion", "codigo", "imagen", "audio", "otro".

EJEMPLO DE SALIDA CORRECTA:
{"title":"Contrato de Arrendamiento - Edificio Colón","summary":"Contrato de arrendamiento firmado el 15/03/2024 entre Inmobiliaria Torres S.A. (arrendador) y Juan Pérez García (arrendatario) para el departamento 4B del Edificio Colón, con renta mensual de $12,500 MXN. Vigencia de 12 meses a partir del 01/04/2024. Incluye cláusulas de depósito ($25,000), mantenimiento y causales de rescisión.","keywords":["arrendamiento","contrato renta","Inmobiliaria Torres","Juan Pérez García","departamento 4B","Edificio Colón","$12500","depósito","renta mensual","rescisión","arrendador","arrendatario","2024"],"semantic_tags":["Legal/Contrato","Finanzas/Arrendamiento","Inmobiliario/Departamento"],"language":"es","content_type_group":"documento"}"#;

        let user_message = format!(
            "Archivo: {}\nCarpeta: {}\n\nContenido del documento:\n{}",
            filename,
            folder_path,
            safe_truncate(content_preview, 1200)
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
                                "Error parseando descripción (directo: {}, extraído: {}). Fallback local.",
                                first_err, second_err
                            );
                            debug!("Respuesta LLM raw ({}ch): {}", response.len(), safe_truncate(&response, 300));
                            Self::fallback_file_description(filename, content_preview)
                        }
                    }
                } else {
                    warn!(
                        "Respuesta describe_file sin JSON ({}). Fallback local.",
                        first_err
                    );
                    debug!("Respuesta LLM raw ({}ch): {}", response.len(), safe_truncate(&response, 300));
                    Self::fallback_file_description(filename, content_preview)
                }
            }
        };

        if desc.title.trim().is_empty() {
            desc.title = normalize_filename_title(filename);
        }
        // Si el LLM no generó summary pero sí description, usar description como summary
        if desc.summary.is_none() || desc.summary.as_deref().map_or(true, |s| s.trim().is_empty()) {
            if !desc.description.trim().is_empty() {
                desc.summary = Some(desc.description.clone());
            } else {
                desc.summary = Some(safe_truncate(content_preview, 200).to_string());
            }
        }
        if desc.description.trim().is_empty() {
            desc.description = desc.summary.clone().unwrap_or_else(|| {
                safe_truncate(content_preview, 220).to_string()
            });
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
            "describe_file: title={:?}, keywords={:?}, tags={:?}",
            desc.title, desc.keywords, desc.semantic_tags
        );

        Ok(desc)
    }

    /// Envía un prompt JSON con opciones de búsqueda (ultra-ligeras).
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
            options: Some(Self::chat_search_runtime_options()),
            keep_alive: None,
            think: Some(false), // JSON: no gastar tokens en thinking
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
        Ok(chat_resp.message.content.trim().to_string())
    }

    /// Mejora una consulta de búsqueda extrayendo keywords semánticas.
    ///
    /// Genera sinónimos y términos relacionados para ampliar la cobertura
    /// del embedding search con nomic-embed-text. Por ejemplo:
    /// - "mi INE" → ["INE", "credencial", "identificación oficial", "elector"]
    /// - "fotos de la playa" → ["playa", "mar", "vacaciones", "foto"]
    pub async fn enhance_search_query(&self, user_query: &str) -> Result<EnhancedQuery> {
        let system_prompt = r#"Eres un asistente de búsqueda de archivos personales.
Dada la consulta del usuario, genera un JSON con:
- "keywords": 3-6 palabras/frases clave para buscar. Incluye la consulta original Y sinónimos/términos relacionados que podrían aparecer en el contenido del archivo. Ejemplo: "mi INE" → ["INE", "credencial", "identificación oficial", "elector", "instituto nacional electoral"]
- "file_types": extensiones probables si es claro (ej: ["pdf","jpg"]), vacío si no es claro
- "hard_type_filter": true SOLO si el usuario pide explícitamente un tipo (ej: "busca el PDF de...")

Solo JSON."#;

        let response = self.chat_json_search(system_prompt, user_query).await?;

        #[derive(Deserialize)]
        struct MinimalQuery {
            #[serde(default, deserialize_with = "deserialize_string_or_vec")]
            keywords: Vec<String>,
            #[serde(default, deserialize_with = "deserialize_string_or_vec")]
            file_types: Vec<String>,
            #[serde(default)]
            hard_type_filter: bool,
        }

        let minimal: MinimalQuery = match serde_json::from_str(&response) {
            Ok(parsed) => parsed,
            Err(first_err) => {
                if let Some(json_block) = extract_json_object(&response) {
                    match serde_json::from_str(json_block) {
                        Ok(parsed) => parsed,
                        Err(e2) => {
                            warn!(
                                "Error parseando consulta mejorada (directo: {}, extraído: {})",
                                first_err, e2
                            );
                            return Err(SoasError::Ollama(format!(
                                "Respuesta no es JSON válido: {}",
                                safe_truncate(&response, 200)
                            )));
                        }
                    }
                } else {
                    warn!(
                        "Error parseando consulta mejorada: {}",
                        first_err
                    );
                    return Err(SoasError::Ollama(format!(
                        "Respuesta no es JSON válido: {}",
                        safe_truncate(&response, 200)
                    )));
                }
            }
        };

        Ok(EnhancedQuery {
            reasoning: String::new(),
            enhanced_query: String::new(),
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
Dado un archivo, sugiere 1-3 categorías donde clasificarlo.
{}
Si ninguna categoría existente aplica, sugiere nuevas.
Responde SOLO con JSON: {{"categories": ["cat1", "cat2"]}}"#,
            categories_list
        );

        let user_message = format!(
            "Archivo: {}\nContenido: {}",
            filename,
            safe_truncate(content_preview, 1000)
        );

        let response = self.chat_json(&system_prompt, &user_message).await?;

        #[derive(Deserialize)]
        struct CategoriesResponse {
            categories: Vec<String>,
        }

        let cats: CategoriesResponse = match serde_json::from_str(&response) {
            Ok(parsed) => parsed,
            Err(first_err) => {
                if let Some(json_block) = extract_json_object(&response) {
                    serde_json::from_str(json_block).map_err(|e| {
                        SoasError::Ollama(format!(
                            "Error parseando categorías (directo: {}, extraído: {})",
                            first_err, e
                        ))
                    })?
                } else {
                    return Err(SoasError::Ollama(format!(
                        "Error parseando categorías: {}",
                        first_err
                    )));
                }
            }
        };

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
    #[serde(default)]
    pub description: String,
    /// Resumen corto para UI (1-3 oraciones específicas)
    #[serde(default)]
    pub summary: Option<String>,
    /// Keywords para búsqueda.
    #[serde(default, deserialize_with = "deserialize_string_or_vec")]
    pub keywords: Vec<String>,
    /// Etiquetas semánticas amplias (temas, conceptos)
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

/// Consulta de búsqueda mejorada por LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedQuery {
    #[serde(default)]
    pub reasoning: String,
    pub enhanced_query: String,
    #[serde(default, deserialize_with = "deserialize_string_or_vec")]
    pub keywords: Vec<String>,
    #[serde(default, deserialize_with = "deserialize_string_or_vec")]
    pub file_types: Vec<String>,
    #[serde(default)]
    pub content_type: String,
    #[serde(default, deserialize_with = "deserialize_string_or_vec")]
    pub date_hints: Vec<String>,
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
                Ok(value
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect())
            } else {
                Ok(vec![value.to_string()])
            }
        }

        fn visit_seq<A: de::SeqAccess<'de>>(
            self,
            mut seq: A,
        ) -> std::result::Result<Vec<String>, A::Error> {
            let mut vec = Vec::new();
            while let Some(val) = seq.next_element::<String>()? {
                vec.push(val);
            }
            Ok(vec)
        }
    }

    deserializer.deserialize_any(StringOrVec)
}
