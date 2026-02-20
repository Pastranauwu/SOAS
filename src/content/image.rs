use crate::content::ExtractedContent;
use crate::embeddings::OllamaClient;
use crate::error::Result;
use base64::Engine;
use std::path::Path;
use tracing::{debug, info, warn};

/// Resolución máxima (ancho o alto) antes de enviar a OCR.
/// Imágenes más grandes se redimensionan para acelerar la inferencia en CPU.
/// 640px prioriza velocidad en CPU, manteniendo suficiente detalle para
/// descripciones visuales útiles y extracción de texto corto.
const MAX_OCR_DIMENSION: u32 = 640;

/// Prepara una imagen para OCR: la redimensiona si es muy grande y la codifica en base64.
/// Esto reduce drásticamente el tiempo de inferencia en CPU (3-10x para fotos grandes).
fn prepare_image_base64(path: &Path) -> Result<String> {
    let img = image::open(path).map_err(|e| {
        crate::error::SoasError::ContentExtraction(format!(
            "No se pudo abrir la imagen {:?}: {}",
            path, e
        ))
    })?;

    let (w, h) = (img.width(), img.height());

    let img = if w > MAX_OCR_DIMENSION || h > MAX_OCR_DIMENSION {
        debug!(
            "Redimensionando imagen {}x{} → max {}px: {:?}",
            w, h, MAX_OCR_DIMENSION, path.file_name()
        );
        img.resize(
            MAX_OCR_DIMENSION,
            MAX_OCR_DIMENSION,
            image::imageops::FilterType::Triangle, // bilinear — rápido y buena calidad
        )
    } else {
        img
    };

    // Codificar como JPEG (más compacto que PNG para el transporte)
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Jpeg)
        .map_err(|e| {
            crate::error::SoasError::ContentExtraction(format!(
                "Error codificando imagen: {}", e
            ))
        })?;

    let encoded = base64::engine::general_purpose::STANDARD.encode(buf.into_inner());
    debug!(
        "Imagen preparada: {}x{} → base64 {} KB",
        img.width(),
        img.height(),
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

    format!(
        "Resumen visual (fallback): archivo de imagen \"{}\" ({}). Nombre semántico: {}.",
        filename, ext, clean_name
    )
}

/// Extrae un resumen visual de imágenes usando qwen3-vl.
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

    info!("   → Paso único: resumen visual con modelo de visión...");
    let start = std::time::Instant::now();

    // Prompt simple y directo: moondream funciona mejor con instrucciones cortas.
    // qwen3-vl también responde bien a prompts concisos.
    // Evitar formatos complejos que modelos ligeros no siguen en CPU.
    let summary_prompt = "Describe this image in Spanish. Include: what it shows, \
                          main elements, and any visible text. Be concise, 2-3 sentences.";

    match ollama.describe_photo(&image_base64, summary_prompt).await {
        Ok(desc) if !desc.trim().is_empty() => {
            // Limpiar artefactos comunes de moondream: "!!!" prefijo
            let trimmed = desc.trim().trim_start_matches('!').trim().to_string();
            let elapsed = start.elapsed();

            if trimmed.is_empty() || is_garbage_vision_output(&trimmed) {
                warn!(
                    "   ⚠️  Resumen visual basura en {:.1}s, usando fallback por nombre",
                    elapsed.as_secs_f64()
                );
                text = build_filename_fallback_summary(path);
                method = "qwen3vl-fallback-garbage";
            } else {
                let preview = &trimmed[..trimmed.len().min(150)];
                info!(
                    "   ✅ Resumen visual: {} chars en {:.1}s — {:?}",
                    trimmed.len(), elapsed.as_secs_f64(), preview
                );
                text = trimmed;
                method = "qwen3vl-summary";
            }
        }
        Ok(_) => {
            let elapsed = start.elapsed();
            info!(
                "   ⚠️  Resumen visual vacío en {:.1}s, usando fallback por nombre",
                elapsed.as_secs_f64()
            );
            text = build_filename_fallback_summary(path);
            method = "qwen3vl-fallback-empty";
        }
        Err(e) => {
            let elapsed = start.elapsed();
            info!(
                "   ❌ Resumen visual falló en {:.1}s: {}. Usando fallback por nombre",
                elapsed.as_secs_f64(), e
            );
            text = build_filename_fallback_summary(path);
            method = "qwen3vl-fallback-error";
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
/// Algunos modelos de visión (especialmente moondream) producen coordenadas
/// de bounding boxes, texto ultra-corto, secuencias repetitivas, o
/// alucinaciones en scripts no-latinos (tailandés, árabe, chino basura).
/// Estas salidas contaminan los embeddings y deben descartarse.
fn is_garbage_vision_output(text: &str) -> bool {
    let t = text.trim();

    // Demasiado corto para ser una descripción útil
    if t.len() < 15 {
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
        let second_half = &t[t.len() / 2..];
        let unique: std::collections::HashSet<char> = second_half.chars().collect();
        if unique.len() <= 4 {
            return true;
        }
    }

    // ── Detección de alucinación Unicode ──────────────────────────────────
    // Moondream a veces genera texto en scripts aleatorios (tailandés, árabe,
    // CJK basura) cuando no puede interpretar la imagen. El contenido real de
    // archivos en computadoras personales está casi siempre en latín o cirílico.
    // Si más del 40% de los caracteres son non-Latin, es alucinación.
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

    // ── Detección de "!!!" prefijo ── moondream a veces antepone "!!!" a su salida
    let clean = t.trim_start_matches('!').trim();
    if clean.len() < 15 {
        return true;
    }

    false
}
