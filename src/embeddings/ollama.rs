use crate::config::OllamaConfig;
use crate::error::{Result, SoasError};
use regex::Regex;
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

/// Elimina tokens de pensamiento de modelos Qwen3 usando regex.
/// Qwen3 genera `<think>...razonamiento...</think>` antes de la respuesta.
/// Usamos regex para capturar TODO el bloque de thinking de forma robusta,
/// incluyendo saltos de línea y caracteres especiales.
fn strip_thinking_tokens(text: &str) -> String {
    let s = text.trim();
    if s.is_empty() {
        return String::new();
    }

    // Regex: captura <think>CUALQUIER COSA</think> (lazy, con dotall para \n)
    // (?s) activa dot-matches-newline
    let re = Regex::new(r"(?s)<think>.*?</think>").unwrap();

    if re.is_match(s) {
        let cleaned = re.replace_all(s, "").trim().to_string();
        if cleaned.is_empty() {
            // Pensó pero la respuesta real está vacía
            warn!(
                "strip_thinking: regex eliminó thinking ({} chars) pero no quedó respuesta",
                s.len()
            );
        }
        return cleaned;
    }

    // Caso truncado: empieza con <think> pero nunca cierra
    if s.starts_with("<think>") {
        warn!(
            "strip_thinking: pensamiento truncado ({} chars), num_predict insuficiente. Inicio: {:?}",
            s.len(),
            &s[..s.len().min(150)]
        );
        return String::new();
    }

    // Sin thinking tokens, devolver tal cual
    s.to_string()
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
    /// Circuit breaker para description_model de visión.
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
    think: Option<bool>,
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
    #[serde(default)]
    thinking: Option<String>,
}

fn extract_answer_from_thinking(thinking: &str) -> String {
    let t = thinking.trim();
    if t.is_empty() {
        return String::new();
    }

    if let Ok(quoted_re) = Regex::new(r#"(?s)\"([^\"\n]{30,600})\""#) {
        if let Some(captures) = quoted_re.captures_iter(t).last() {
            if let Some(m) = captures.get(1) {
                let candidate = m.as_str().trim();
                if !candidate.is_empty() {
                    return candidate.to_string();
                }
            }
        }
    }

    if let Ok(two_sentences_es) = Regex::new(
        r"(?s)([A-ZÁÉÍÓÚÑ][^\n]{25,260}[\.!?]\s+[A-ZÁÉÍÓÚÑ][^\n]{25,260}[\.!?])",
    ) {
        if let Some(captures) = two_sentences_es.captures_iter(t).last() {
            if let Some(m) = captures.get(1) {
                let candidate = m.as_str().trim();
                if !candidate.is_empty() {
                    return candidate.to_string();
                }
            }
        }
    }

    for marker in ["Possible translation:", "Final answer:", "Respuesta final:"] {
        if let Some(pos) = t.rfind(marker) {
            let after = t[pos + marker.len()..].trim().trim_matches('"').trim();
            if after.len() >= 20 {
                return after.to_string();
            }
        }
    }

    let lines: Vec<&str> = t
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();

    let invalid_markers = [
        "documento|foto|captura|gráfico|otro",
        "si/no + resumen corto",
        "texto visible: si/no",
        "tema: frase corta",
        "is a template",
        "placeholder",
        "fill in",
        "formato esperado:",
    ];

    for line in lines.iter().rev() {
        let lower = line.to_lowercase();
        
        // Skip template echoes
        if invalid_markers.iter().any(|m| lower.contains(m)) {
            continue;
        }
        
        let looks_like_meta = lower.starts_with("wait")
            || lower.starts_with("let me")
            || lower.starts_with("i should")
            || lower.starts_with("i need")
            || lower.starts_with("the user")
            || lower.starts_with("maybe")
            || lower.starts_with("so:")
            || lower.starts_with("hmm")
            || lower.starts_with("ok,")
            || lower.starts_with("first,")
            || lower.starts_with("now,");

        if !looks_like_meta && line.len() >= 30 {
            // Additional check: valid sentence structure?
            // Should not end with colon unless it's a list header we want
            if line.ends_with(':') && line.len() < 50 {
                continue;
            }
            return (*line).to_string();
        }
    }

    String::new()
}

fn extract_visible_response(message: &ChatResponseMessage) -> String {
    let content = strip_thinking_tokens(&message.content);
    if !content.trim().is_empty() {
        return content;
    }

    if let Some(thinking) = &message.thinking {
        let recovered = extract_answer_from_thinking(thinking);
        if !recovered.is_empty() {
            warn!(
                "Respuesta en content vacía; recuperando texto desde campo thinking ({} chars)",
                thinking.len()
            );
            return recovered;
        }
    }

    String::new()
}

fn vision_debug_enabled() -> bool {
    matches!(
        std::env::var("SOAS_VISION_DEBUG"),
        Ok(v) if v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes")
    )
}

fn looks_like_useful_vision_text(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.len() < 30 {
        return false;
    }

    let lower = trimmed.to_lowercase();

    // Eco de plantilla: no aporta semántica real.
    let template_markers = [
        "documento|foto|captura|gráfico|otro",
        "si/no + resumen corto",
        "frase corta",
        "texto visible: si/no",
        "tipo: documento|foto",
    ];
    if template_markers.iter().any(|m| lower.contains(m)) {
        return false;
    }

    // Patrones de texto meta/residual que no describen realmente la imagen.
    let bad_markers = [
        "or something similar",
        "need to check",
        "the image has",
        "the text visible",
        "is right",
        "it appears",
        "let me",
        "i should",
        "i need",
        "maybe",
        "final answer",
        "possible translation",
    ];

    if bad_markers.iter().any(|m| lower.contains(m)) {
        return false;
    }

    // Formato esperado en modo rápido: Tipo / Texto visible / Tema.
    let has_structured = lower.contains("tipo:")
        && lower.contains("texto visible:")
        && lower.contains("tema:");

    if has_structured {
        let mut good_fields = 0usize;
        for line in trimmed.lines().map(|l| l.trim()).filter(|l| !l.is_empty()) {
            let lower_line = line.to_lowercase();
            let Some((_, payload_raw)) = line.split_once(':') else { continue; };
            let payload = payload_raw.trim();
            let payload_lower = payload.to_lowercase();

            let is_structured_field = lower_line.starts_with("tipo:")
                || lower_line.starts_with("texto visible:")
                || lower_line.starts_with("tema:");
            if !is_structured_field {
                continue;
            }

            let payload_invalid = payload.is_empty()
                || payload.contains('[')
                || payload.contains(']')
                || payload.contains('|')
                || payload.contains('+')
                || payload_lower.contains("si/no")
                || payload_lower.contains("resumen corto")
                || payload_lower.contains("frase corta")
                || payload_lower.contains("documento|foto")
                || payload.len() < 4;

            if !payload_invalid {
                good_fields += 1;
            }
        }

        // Requerir al menos 2 campos útiles para aceptar estructurado.
        if good_fields < 2 {
            return false;
        }
    }

    // Aceptar también descripciones libres en español con señales mínimas.
    let has_spanish_signal = [
        "imagen",
        "foto",
        "documento",
        "texto",
        "visible",
        "tema",
        "muestra",
    ]
    .iter()
    .any(|w| lower.contains(w));

    has_structured || has_spanish_signal
}

impl OllamaClient {
    fn vision_runtime_options(model: &str) -> serde_json::Value {
        let model_lower = model.to_lowercase();

        if model_lower.contains("qwen3-vl") || model_lower.contains("qwen2.5-vl") {
            return serde_json::json!({
                "temperature": 0.1,
                "num_ctx": 4096,
                "num_predict": 1024
            });
        }
        
        if model_lower.contains("phi") {
            // LLaVA-W / Phi-3 ajustes
            // num_predict=512: el límite anterior de 256 truncaba el texto visible
            // de documentos (credenciales, recibos). 512 da margen suficiente
            // sin impacto notable en tiempo cuando la respuesta es corta.
            return serde_json::json!({
                "temperature": 0.1,
                "num_ctx": 3072,
                "num_predict": 512
            });
        }

        serde_json::json!({
            "temperature": 0.1,
            "num_ctx": 2048,
            "num_predict": 512
        })
    }

    fn chat_runtime_options() -> serde_json::Value {
        serde_json::json!({
            "num_ctx": 3072,
            // 384 tokens: qwen3 gasta ~150-250 en  groundwork incluso con
            // instrucciones de no pensar. 384 deja margen para respuesta.
            "num_predict": 512,
            "temperature": 0.2
        })
    }

    fn chat_json_runtime_options() -> serde_json::Value {
        serde_json::json!({
            "num_ctx": 3072,
            // 1024 tokens: qwen3 piensa ~200-300 tokens + JSON respuesta ~200-400.
            // Con 512 se truncaba el JSON consistentemente.
            "num_predict": 1024,
            "temperature": 0.3
        })
    }

    /// Opciones para búsqueda interactiva.
    /// Contexto pequeño porque la query es corta (~10-30 tokens).
    /// num_predict=512: qwen3 piensa ~200 tokens + JSON respuesta ~80.
    /// temperature=0: respuestas deterministas para la misma consulta.
    fn chat_search_runtime_options() -> serde_json::Value {
        serde_json::json!({
            "num_ctx": 2048,
            "num_predict": 512,
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
            think: Some(false),
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
        Ok(extract_visible_response(&chat_resp.message))
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
            think: Some(false),
            // NO usar format:"json" con qwen3 — la generación restringida
            // por gramática impide <think> tags y el modelo produce respuesta vacía.
            // Nuestro parser de 3 niveles (JSON directo → bloque extraído → fallback)
            // maneja la extracción del JSON desde texto libre.
            format: None,
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
        Ok(extract_visible_response(&chat_resp.message))
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

    /// Describe una foto/ilustración usando el modelo de descripción visual.
    /// Se usa como fallback cuando OCR no encuentra texto suficiente.
    /// Incluye circuit breaker: si falla 2 veces consecutivas, se desactiva.
    pub async fn describe_photo(&self, image_base64: &str, prompt: &str) -> Result<String> {
        // Circuit breaker: si ya falló 3+ veces consecutivas, no intentar.
        // Umbral de 3 (no 2) porque la primera imagen paga la carga del modelo
        // en RAM (~10-15s en CPU) que puede causar 1-2 timeouts transitorios.
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
                    if count >= 4 {
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
                    if count >= 3 {
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
            think: Some(false),
            format: None,
            // Perfil rápido por defecto para indexación de imágenes:
            // - num_predict bajo (300) porque solo necesitamos metadata breve
            // - temperature baja para consistencia
            // - num_ctx moderado para evitar salidas vacías en qwen3-vl
            options: Some(Self::vision_runtime_options(model)),
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
        let raw_content = &chat_resp.message.content;
        let raw_thinking = chat_resp.message.thinking.as_deref().unwrap_or("");

        if vision_debug_enabled() {
            println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!(
                "🔬 RAW de [{}]: content={} chars | thinking={} chars",
                model,
                raw_content.len(),
                raw_thinking.len()
            );
            println!("─── contenido ───");
            println!("{}", &raw_content[..raw_content.len().min(800)]);
            if !raw_thinking.is_empty() {
                println!("─── thinking ───");
                println!("{}", &raw_thinking[..raw_thinking.len().min(800)]);
            }
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        }

        let result = extract_visible_response(&chat_resp.message);

        if result.is_empty() && !raw_content.trim().is_empty() {
            warn!(
                "🔬 Respuesta perdida por strip_thinking: el modelo generó {} chars \
                 pero después de limpiar thinking quedó vacío. \
                 Esto indica que num_predict es insuficiente para este modelo.",
                raw_content.len()
            );
        }

        if !result.is_empty() {
            if !looks_like_useful_vision_text(&result) {
                warn!(
                    "Respuesta de visión descartada por baja calidad/meta-residual: {:?}",
                    &result[..result.len().min(180)]
                );
                return Ok(String::new());
            }
            info!("🔬 Resultado limpio: {:?}", &result[..result.len().min(200)]);
        }

        Ok(result)
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
        folder_path: &str,
    ) -> Result<FileDescription> {
        let system_prompt = "Genera SOLO JSON válido en español. Sin explicaciones, sin markdown, sin texto fuera del JSON.";

        // Incluir ruta de carpeta como contexto — ayuda al modelo a entender
        // el dominio del archivo (ej: "Descargas/trabajo/" vs "Descargas/fotos/")
        let user_message = format!(
            "Genera JSON con estos campos para el archivo:\n\
             - \"title\": título corto descriptivo\n\
             - \"description\": resumen semántico (1 oración)\n\
             - \"keywords\": 4-8 palabras clave\n\
             - \"semantic_tags\": 2-4 temas\n\
             - \"language\": \"es\" | \"en\" | etc.\n\
             - \"content_type_group\": \"documento\" | \"imagen\" | \"hoja_calculo\" | \"codigo\" | \"presentacion\" | \"comprimido\" | \"archivo\"\n\
             \nArchivo: {}\nCarpeta: {}\n\nContenido (extracto):\n{}",
            filename,
            folder_path,
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
                            info!("Respuesta LLM raw ({}ch): {}", response.len(), safe_truncate(&response, 300));
                            Self::fallback_file_description(filename, content_preview)
                        }
                    }
                } else {
                    warn!(
                        "Respuesta describe_file sin JSON utilizable ({}). Usando fallback local.",
                        first_err
                    );
                    info!("Respuesta LLM raw ({}ch): {}", response.len(), safe_truncate(&response, 300));
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
            think: Some(false),
            // Sin format:"json" — misma razón que chat_json().
            format: None,
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
        Ok(extract_visible_response(&chat_resp.message))
    }

    /// Mejora una consulta de búsqueda extrayendo keywords y tipo de archivo.
    ///
    /// Prompt ultra-compacto: ~150 tokens de input vs ~600 del anterior.
    /// Genera ~50-80 tokens de output. En CPU con num_ctx=1024, tarda ~3-5s vs ~16s.
    pub async fn enhance_search_query(&self, user_query: &str) -> Result<EnhancedQuery> {
        let system_prompt = "Eres un extractor de intención de búsqueda de archivos personales (documentos, fotos, PDFs). \
                            De la consulta del usuario, extrae un JSON con:\n\
                            - \"keywords\": 2-5 palabras/frases clave sustantivas (incluir sinónimos útiles, ej: \"INE\" → [\"INE\", \"credencial\", \"identificación\"])\n\
                            - \"file_types\": extensiones probables (vacío si no es claro)\n\
                            - \"hard_type_filter\": true solo si pide explícitamente un tipo de archivo\n\
                            Solo JSON, sin explicaciones.";

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

        // Parseo con fallback: directo → extraer bloque JSON → error
        let minimal: MinimalQuery = match serde_json::from_str(&response) {
            Ok(parsed) => parsed,
            Err(first_err) => {
                // qwen3 puede envolver JSON en texto/pensamiento residual
                if let Some(json_block) = extract_json_object(&response) {
                    match serde_json::from_str(json_block) {
                        Ok(parsed) => parsed,
                        Err(e2) => {
                            warn!("Error parseando consulta mejorada (directo: {}, extraído: {}) | raw ({}ch): {}",
                                first_err, e2, response.len(), safe_truncate(&response, 200));
                            return Err(SoasError::Ollama(format!(
                                "Respuesta no es JSON válido: {}",
                                safe_truncate(&response, 200)
                            )));
                        }
                    }
                } else {
                    warn!("Error parseando consulta mejorada: {} | raw ({}ch): {}",
                        first_err, response.len(), safe_truncate(&response, 200));
                    return Err(SoasError::Ollama(format!(
                        "Respuesta no es JSON válido: {}",
                        safe_truncate(&response, 200)
                    )));
                }
            }
        };

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
    /// a veces devuelve un string en vez de array, se maneja con deserialize_string_or_vec.
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
                // a veces devuelve "word1, word2, word3" en vez de ["word1","word2","word3"]
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
