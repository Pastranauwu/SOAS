use crate::content::ExtractedContent;
use crate::error::{Result, SoasError};
use std::path::Path;
use tracing::{debug, warn};

/// Extrae texto de archivos PDF
pub fn extract(path: &Path) -> Result<ExtractedContent> {
    debug!("Extrayendo texto de PDF: {:?}", path);

    let text = pdf_extract::extract_text(path).map_err(|e| {
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
