use crate::content::ExtractedContent;
use crate::embeddings::OllamaClient;
use crate::error::Result;
use base64::Engine;
use std::path::Path;
use tracing::{debug, info, warn};

/// Limpia la salida del VLM eliminando frases de plantilla que contaminan
/// keywords y embeddings. Estas frases aparecen en casi toda imagen y
/// no aportan información semántica diferenciadora.
fn clean_vision_output(raw: &str) -> String {
    let mut lines: Vec<&str> = Vec::new();

    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Quitar numeración de lista: "1. ", "2. ", "3. "
        let stripped = if trimmed.len() > 3
            && trimmed.as_bytes()[0].is_ascii_digit()
            && trimmed.as_bytes()[1] == b'.'
            && trimmed.as_bytes()[2] == b' '
        {
            &trimmed[3..]
        } else {
            trimmed
        };

        let lower = stripped.to_lowercase();

        // Descartar frases de plantilla vacías de contenido
        let is_template = lower.contains("no hay texto legible")
            || lower.contains("no hay texto visible")
            || lower.contains("no se puede leer")
            || lower.contains("no contiene texto")
            || lower.starts_with("esta es una imagen en formato")
            || lower.starts_with("la imagen es una fotografía")
            || lower.starts_with("esta es una imagen de formato")
            || (lower.starts_with("la imagen es un") && lower.len() < 40);

        if is_template {
            continue;
        }

        // Quitar prefijos redundantes tipo "Tema: ", "Texto visible: "
        // Mantener el contenido después del prefijo
        let content = if let Some(after) = lower.strip_prefix("tema: ") {
            let _ = after; // solo para que compile
            stripped.splitn(2, ':').nth(1).map(|s| s.trim()).unwrap_or(stripped)
        } else if let Some(after) = lower.strip_prefix("texto visible: ") {
            let _ = after;
            stripped.splitn(2, ':').nth(1).map(|s| s.trim()).unwrap_or(stripped)
        } else {
            stripped
        };

        if !content.is_empty() {
            lines.push(content);
        }
    }

    let result = lines.join(". ");
    // Si quedó vacío después de limpiar, devolver el raw original
    if result.trim().is_empty() {
        raw.trim().to_string()
    } else {
        result
    }
}

/// Resolución máxima para enviar al VLM.
///
/// Lección aprendida: llava-phi3 (~3.8B params) NO puede hacer OCR real.
/// A 1280px tarda 100s+ y sigue alucinando ("CURP argentino" para una INE).
/// A 512px no alcanza a ver detalles para clasificar correctamente.
///
/// Punto óptimo: 768px
///   - Suficiente para que el VLM clasifique (documento vs foto vs screenshot)
///   - Suficiente para describir contenido visual general
///   - ~15-40s por imagen (aceptable para indexación batch)
///   - NO confiar en transcripción de texto (usar filename para eso)
const MAX_VLM_DIMENSION: u32 = 768;

/// Prepara una imagen para el VLM: redimensiona a max 768px y codifica en base64.
fn prepare_image_base64(path: &Path) -> Result<String> {
    let img = image::open(path).map_err(|e| {
        crate::error::SoasError::ContentExtraction(format!(
            "No se pudo abrir la imagen {:?}: {}",
            path, e
        ))
    })?;

    let (w, h) = (img.width(), img.height());

    let img = if w > MAX_VLM_DIMENSION || h > MAX_VLM_DIMENSION {
        debug!(
            "Redimensionando imagen {}x{} → max {}px: {:?}",
            w, h, MAX_VLM_DIMENSION, path.file_name()
        );
        img.resize(
            MAX_VLM_DIMENSION,
            MAX_VLM_DIMENSION,
            image::imageops::FilterType::Lanczos3,
        )
    } else {
        img
    };

    // PNG lossless: preserva bordes mejor que JPEG a resolución reducida.
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png)
        .map_err(|e| {
            crate::error::SoasError::ContentExtraction(format!(
                "Error codificando imagen: {}", e
            ))
        })?;

    let encoded = base64::engine::general_purpose::STANDARD.encode(buf.into_inner());
    info!(
        "   📐 Imagen: {}x{} → {}x{} PNG {} KB",
        w, h,
        img.width(), img.height(),
        encoded.len() / 1024
    );
    Ok(encoded)
}

fn build_filename_fallback_summary(path: &Path) -> String {
    let filename = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "imagen".to_string());

    let clean_name = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "imagen".to_string())
        .replace(['_', '-', '.'], " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");

    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_else(|| "img".to_string());

    // Generar una descripción semántica más rica a partir del nombre.
    // Los nombres de archivo suelen ser bastante descriptivos:
    // "ine_frente.jpeg" → "Imagen: ine frente. Podría ser credencial INE (frente).
    // Esto ayuda al embedding a capturar la intención de búsqueda.
    let semantic_hint = infer_semantic_hint_from_name(&clean_name);

    if semantic_hint.is_empty() {
        format!(
            "Imagen: {}. Archivo de imagen {} ({}).",
            clean_name, filename, ext
        )
    } else {
        format!(
            "Imagen: {}. {}. Archivo {} ({}).",
            clean_name, semantic_hint, filename, ext
        )
    }
}

fn build_filename_fallback_summary_with_hint(path: &Path, hint: &str) -> String {
    let base = build_filename_fallback_summary(path);
    if hint.trim().is_empty() {
        return base;
    }

    format!("{} Pista visual: {}.", base.trim_end_matches('.'), hint.trim())
}

fn extract_structured_visual_hint(text: &str) -> Option<String> {
    let cleaned = text.trim().replace('\r', "");
    if cleaned.is_empty() {
        return None;
    }

    let mut parts = Vec::new();
    for line in cleaned.lines().map(|l| l.trim()).filter(|l| !l.is_empty()) {
        let lower = line.to_lowercase();
        let is_structured = lower.starts_with("tema:") || lower.starts_with("texto visible:");
        if !is_structured {
            continue;
        }

        let payload = line
            .split_once(':')
            .map(|(_, right)| right.trim())
            .unwrap_or("");
        let payload_lower = payload.to_lowercase();

        let is_template_echo = payload_lower.is_empty()
            || payload.contains('[')
            || payload.contains(']')
            || payload.contains('|')
            || payload.contains('+')
            || payload_lower.contains("si/no")
            || payload_lower.contains("resumen corto")
            || payload_lower.contains("frase corta")
            || payload_lower.contains("documento") && payload_lower.contains("foto")
            || payload_lower.contains("captura") && payload_lower.contains("gráfico")
            || payload.len() < 6;

        if is_template_echo {
            continue;
        }

        parts.push(format!("{}: {}", line.split_once(':').map(|(k, _)| k.trim()).unwrap_or("Dato"), payload));
    }

    if parts.is_empty() {
        let lower = cleaned.to_lowercase();
        if lower.starts_with("tema:") || lower.starts_with("texto visible:") {
            return Some(cleaned.to_string());
        }
        return None;
    }

    Some(parts.join(" | "))
}

/// Infiere una pista semántica del nombre de archivo para mejorar el embedding.
/// Reconoce patrones comunes en nombres de archivos de usuario.
fn infer_semantic_hint_from_name(clean_name: &str) -> String {
    let lower = clean_name.to_lowercase();
    let mut hints = Vec::new();

    // Documentos de identidad
    if lower.contains("ine") || lower.contains("credencial") || lower.contains("identificacion") {
        hints.push("credencial de identificación INE");
    }
    if lower.contains("pasaporte") {
        hints.push("documento de pasaporte");
    }
    if lower.contains("curp") {
        hints.push("documento CURP");
    }
    if lower.contains("rfc") {
        hints.push("documento RFC");
    }
    if lower.contains("licencia") {
        hints.push("licencia de conducir");
    }
    if lower.contains("acta") && lower.contains("nacimiento") {
        hints.push("acta de nacimiento");
    }

    // Orientación de la imagen
    if lower.contains("frente") || lower.contains("frontal") || lower.contains("front") {
        hints.push("vista frontal");
    }
    if lower.contains("posterior") || lower.contains("atras") || lower.contains("reverso") || lower.contains("back") {
        hints.push("vista posterior/reverso");
    }

    // Fotos personales
    if lower.contains("foto") || lower.contains("selfie") || lower.contains("retrato") {
        hints.push("fotografía personal");
    }
    if lower.contains("recibo") || lower.contains("comprobante") {
        hints.push("recibo o comprobante");
    }
    if lower.contains("factura") {
        hints.push("factura");
    }
    if lower.contains("constancia") {
        hints.push("constancia o certificado");
    }
    if lower.contains("diploma") || lower.contains("titulo") || lower.contains("certificado") {
        hints.push("diploma o certificado");
    }

    if hints.is_empty() {
        String::new()
    } else {
        format!("Posible contenido: {}", hints.join(", "))
    }
}

/// Extrae un resumen visual de imágenes usando modelos VLM ligeros (phi-3/qwen).
///
/// En CPU se evita OCR como paso separado para no duplicar inferencia.
/// En su lugar se realiza una sola llamada que pide:
/// - resumen semántico breve
/// - elementos visuales principales
/// - texto visible relevante (si existe)
pub async fn extract(path: &Path, ollama: &OllamaClient) -> Result<ExtractedContent> {
    let fname = path.file_name().unwrap_or_default().to_string_lossy();
    info!("🖼️  Procesando imagen: {}", fname);

    // Redimensionar si es necesario + codificar en base64 (JPEG comprimido)
    let image_base64 = prepare_image_base64(path)?;

    let text;
    let method;
    let mut vision_hint: Option<String> = None;

    info!("   → Paso único: resumen visual con modelo de visión...");
    let start = std::time::Instant::now();

    // Prompt SIMPLE y directo. Lección aprendida:
    // - llava-phi3 NO puede hacer OCR (alucina texto inexistente)
    // - Pedir transcripción produce peores resultados que pedir descripción
    // - El nombre del archivo es MÁS confiable que el VLM para identificar docs
    // - El VLM solo aporta: clasificación visual + descripción de contenido
    let summary_prompt = "Describe esta imagen en español en 2-3 oraciones. \
        ¿Qué tipo de imagen es? ¿Qué muestra? \
        Si parece un documento oficial, di de qué tipo parece ser.";

    // ── Retry con backoff para tolerancia a fallos transitorios ──────────
    // La primera imagen puede fallar por carga del modelo en RAM.
    // Reintentar 1 vez con pausa permite que Ollama termine la carga.
    let max_retries = 1;
    let mut last_error: Option<String> = None;
    let mut vision_result = None;

    for attempt in 0..=max_retries {
        if attempt > 0 {
            info!("   🔄 Reintentando visión (intento {}/{}), esperando 3s para carga de modelo...",
                  attempt + 1, max_retries + 1);
            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        }

        match ollama.describe_photo(&image_base64, summary_prompt).await {
            Ok(desc) => {
                vision_result = Some(desc);
                break;
            }
            Err(e) => {
                let elapsed = start.elapsed();
                warn!(
                    "   ⚠️  Intento {} falló en {:.1}s: {}",
                    attempt + 1, elapsed.as_secs_f64(), e
                );
                last_error = Some(e.to_string());
            }
        }
    }

    match vision_result {
        Some(desc) if !desc.trim().is_empty() => {
            // Limpiar artefactos de VLMs ("!!!" prefijo) + frases de plantilla
            let raw_trimmed = desc.trim().trim_start_matches('!').trim();
            let trimmed = clean_vision_output(raw_trimmed);
            let elapsed = start.elapsed();

            if trimmed.is_empty() || is_garbage_vision_output(&trimmed) {
                warn!(
                    "   ⚠️  Resumen visual basura en {:.1}s, usando fallback por nombre",
                    elapsed.as_secs_f64()
                );
                if let Some(hint) = extract_structured_visual_hint(&trimmed) {
                    vision_hint = Some(hint.clone());
                    text = build_filename_fallback_summary_with_hint(path, &hint);
                    method = "vision-fallback-hint";
                } else {
                    text = build_filename_fallback_summary(path);
                    method = "vision-fallback-garbage";
                }
            } else {
                // safe_preview: corte en frontera de char UTF-8
                let end = {
                    let mut e = trimmed.len().min(150);
                    while e > 0 && !trimmed.is_char_boundary(e) {
                        e -= 1;
                    }
                    e
                };
                let preview = &trimmed[..end];
                info!(
                    "   ✅ Resumen visual: {} chars en {:.1}s — {:?}",
                    trimmed.len(), elapsed.as_secs_f64(), preview
                );
                text = trimmed;
                method = "vision-summary";
            }
        }
        Some(_) => {
            let elapsed = start.elapsed();
            info!(
                "   ⚠️  Resumen visual vacío en {:.1}s, usando fallback por nombre",
                elapsed.as_secs_f64()
            );
            text = build_filename_fallback_summary(path);
            method = "vision-fallback-empty";
        }
        None => {
            let elapsed = start.elapsed();
            info!(
                "   ❌ Resumen visual falló en {:.1}s: {}. Usando fallback por nombre",
                elapsed.as_secs_f64(), last_error.as_deref().unwrap_or("unknown")
            );
            text = build_filename_fallback_summary(path);
            method = "vision-fallback-error";
        }
    }

    info!(
        "   📝 Resultado: método={}, texto={} chars para {}",
        method, text.len(), fname
    );

    let mut extra = std::collections::HashMap::new();
    extra.insert("type".to_string(), "image".to_string());
    extra.insert("ocr".to_string(), method.to_string());
    extra.insert("vision".to_string(), method.to_string());
    if let Some(hint) = vision_hint {
        extra.insert("vision_hint".to_string(), hint);
    }

    Ok(ExtractedContent {
        text,
        title: path.file_stem().map(|s| s.to_string_lossy().to_string()),
        author: None,
        page_count: None,
        extra,
    })
}

/// Detecta salidas basura de VLMs que no son descripciones útiles.
///
/// Algunos modelos de visión producen coordenadas de bounding boxes,
/// texto ultra-corto, secuencias repetitivas, o alucinaciones en
/// scripts no-latinos. Estas salidas contaminan los embeddings.
fn is_garbage_vision_output(text: &str) -> bool {
    let t = text.trim();
    let lower = t.to_lowercase();

    // Detectar eco de plantilla: el VLM copió las instrucciones del prompt
    let has_template_echo = lower.contains("[")
        || lower.contains("]")
        || lower.contains("si/no + resumen corto")
        || lower.contains("documento|foto|captura")
        || lower.contains("1 frase corta")
        || lower.contains("texto visible: si/no")
        || lower.contains("tema: frase corta")
        || lower.contains("2-3 oraciones");
    if has_template_echo {
        return true;
    }

    // Demasiado corto para ser una descripción útil.
    if t.len() < 40 {
        return true;
    }

    // Meta-razonamiento interno del LLM: no describe la imagen.
    let meta_markers = [
        "let me",
        "i need",
        "the user",
        "wait,",
        "include: what it shows",
        "main elements",
        "visible text",
        "possible translation",
        "final answer",
    ];
    if meta_markers.iter().any(|m| lower.contains(m)) {
        return true;
    }

    // Coordenadas de bounding box: [0.42, 0.22, 0.69, 0.39]
    if t.starts_with('[') && t.ends_with(']') {
        return true;
    }

    // Patrón de path-like que no es descripción: "ids/certificates/credentials"
    if t.len() < 40 && t.contains('/') && !t.contains(' ') {
        return true;
    }

    // Secuencias repetitivas (ej: "idmex18372729292929292929...")
    // Verificar si la segunda mitad del texto tiene muy poca variedad de caracteres
    if t.len() > 40 {
        let mid = t.len() / 2;
        // Ajustar al char boundary más cercano para evitar panic con UTF-8
        let safe_mid = (mid..t.len()).find(|&i| t.is_char_boundary(i)).unwrap_or(t.len());
        let second_half = &t[safe_mid..];
        let unique: std::collections::HashSet<char> = second_half.chars().collect();
        if unique.len() <= 4 {
            return true;
        }
    }

    // ── Detección de palabras repetitivas ──────────────────────────────
    // VLMs pueden generar secuencias repetitivas que parecen texto válido.
    // Si la diversidad de palabras es muy baja (< 30%), es basura.
    {
        let words: Vec<&str> = t.split_whitespace().collect();
        if words.len() >= 6 {
            let unique_words: std::collections::HashSet<&str> = words.iter()
                .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
                .filter(|w| !w.is_empty())
                .collect();
            let diversity = unique_words.len() as f32 / words.len() as f32;
            if diversity < 0.30 {
                return true;
            }
        }
    }

    // ── Detección de alucinación Unicode ──────────────────────────────────
    // VLMs pueden generar texto en scripts aleatorios cuando no pueden
    // interpretar la imagen. Si más del 40% son non-Latin, es alucinación.
    let total_chars = t.chars().count();
    if total_chars > 0 {
        let non_latin = t.chars().filter(|c| {
            // Aceptar: ASCII, Latin Extended, acentos, diacríticos, puntuación, números
            // Rechazar: Thai, Arabic, CJK, Devanagari, etc.
            !c.is_ascii()
                && !matches!(*c as u32,
                    0x00C0..=0x024F    // Latin Extended-A/B (ñ, ü, etc.)
                    | 0x0300..=0x036F  // Combining diacritical marks
                    | 0x1E00..=0x1EFF  // Latin Extended Additional
                    | 0x0400..=0x04FF  // Cyrillic (por si acaso)
                    | 0x2000..=0x22FF  // General punctuation, math, currency
                )
        }).count();

        let ratio = non_latin as f32 / total_chars as f32;
        if ratio > 0.40 {
            return true;
        }
    }

    // ── Detección de "!!!" prefijo ── algunos VLMs anteponen "!!!" a su salida
    let clean = t.trim_start_matches('!').trim();
    if clean.len() < 40 {
        return true;
    }

    false
}
