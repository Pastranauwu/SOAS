use crate::content::ExtractedContent;
use crate::error::{Result, SoasError};
use std::panic;
use std::path::Path;
use tracing::{debug, warn};

/// Extrae texto de archivos PDF.
///
/// Usa `catch_unwind` para atrapar panics internos de `pdf-extract`
/// (por ejemplo, color spaces no soportados como DeviceN/PANTONE)
/// y convertirlos en errores recuperables en lugar de abortar el proceso.
pub fn extract(path: &Path) -> Result<ExtractedContent> {
    debug!("Extrayendo texto de PDF: {:?}", path);

    let path_owned = path.to_path_buf();
    let text = panic::catch_unwind(move || {
        pdf_extract::extract_text(&path_owned)
    })
    .map_err(|panic_info| {
        let msg = if let Some(s) = panic_info.downcast_ref::<String>() {
            s.clone()
        } else if let Some(s) = panic_info.downcast_ref::<&str>() {
            s.to_string()
        } else {
            "panic desconocido en pdf-extract".to_string()
        };
        warn!("pdf-extract hizo panic en {:?}: {}", path, msg);
        SoasError::ContentExtraction(format!("PDF panic (color space u otro): {}", msg))
    })?
    .map_err(|e| {
        warn!("Error extrayendo texto de PDF {:?}: {}", path, e);
        SoasError::ContentExtraction(format!("Error en PDF: {}", e))
    })?;

    // Limpiar el texto extraído
    let clean_text = clean_pdf_text(&text);

    Ok(ExtractedContent {
        text: clean_text,
        title: None, // PDF metadata se podría extraer con lopdf si se necesita
        author: None,
        page_count: None,
        extra: std::collections::HashMap::new(),
    })
}

/// Limpia texto extraído de PDF (elimina saltos de línea excesivos, etc.)
fn clean_pdf_text(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_was_newline = false;

    for ch in text.chars() {
        if ch == '\n' || ch == '\r' {
            if !prev_was_newline {
                result.push('\n');
                prev_was_newline = true;
            }
        } else {
            prev_was_newline = false;
            result.push(ch);
        }
    }

    result.trim().to_string()
}
