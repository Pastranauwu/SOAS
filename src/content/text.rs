use crate::content::ExtractedContent;
use crate::error::Result;
use encoding_rs::WINDOWS_1252;
use std::path::Path;
use tracing::debug;

/// Extrae contenido de archivos de texto plano
pub fn extract(path: &Path) -> Result<ExtractedContent> {
    debug!("Extrayendo texto de: {:?}", path);

    // Intentar leer como UTF-8 primero
    let text = match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(_) => {
            // Intentar con encoding Windows-1252 / Latin-1
            let bytes = std::fs::read(path)?;
            let (decoded, _, had_errors) = WINDOWS_1252.decode(&bytes);
            if had_errors {
                // Forzar UTF-8 con reemplazo de caracteres inválidos
                String::from_utf8_lossy(&bytes).to_string()
            } else {
                decoded.to_string()
            }
        }
    };

    let title = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string());

    Ok(ExtractedContent {
        text,
        title,
        author: None,
        page_count: None,
        extra: std::collections::HashMap::new(),
    })
}
