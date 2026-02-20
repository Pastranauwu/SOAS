pub mod csv_extractor;
pub mod docx;
pub mod image;
pub mod pdf;
pub mod text;

use crate::embeddings::OllamaClient;
use crate::error::{Result, SoasError};
use std::path::Path;
use tracing::debug;

/// Resultado de la extracción de contenido
#[derive(Debug, Clone)]
pub struct ExtractedContent {
    /// Texto extraído del archivo
    pub text: String,
    /// Título detectado (si aplica)
    pub title: Option<String>,
    /// Autor detectado (si aplica)
    pub author: Option<String>,
    /// Número de páginas (si aplica)
    pub page_count: Option<u32>,
    /// Metadatos adicionales
    pub extra: std::collections::HashMap<String, String>,
}

impl ExtractedContent {
    pub fn empty() -> Self {
        Self {
            text: String::new(),
            title: None,
            author: None,
            page_count: None,
            extra: std::collections::HashMap::new(),
        }
    }

    /// Recorta el texto a una longitud máxima
    pub fn truncated_text(&self, max_len: usize) -> &str {
        if self.text.len() <= max_len {
            &self.text
        } else {
            // Encontrar un punto de corte seguro (no cortar UTF-8)
            let mut end = max_len;
            while !self.text.is_char_boundary(end) && end > 0 {
                end -= 1;
            }
            &self.text[..end]
        }
    }
}

/// Extrae contenido de un archivo según su extensión
pub async fn extract_content(
    path: &Path,
    ollama: Option<&OllamaClient>,
) -> Result<ExtractedContent> {
    let extension = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    debug!("Extrayendo contenido de {:?} (ext: {})", path, extension);

    match extension.as_str() {
        // Texto plano
        "txt" | "md" | "markdown" | "rst" | "log" => text::extract(path),

        // Código fuente (se trata como texto)
        "rs" | "py" | "js" | "ts" | "jsx" | "tsx" | "html" | "css" | "scss" | "sass" | "go"
        | "java" | "c" | "cpp" | "h" | "hpp" | "cs" | "rb" | "php" | "swift" | "kt"
        | "scala" | "r" | "sql" | "sh" | "bash" | "zsh" | "fish" | "ps1" | "bat" | "cmd" => {
            text::extract(path)
        }

        // Configuración (como texto)
        "json" | "yaml" | "yml" | "toml" | "ini" | "cfg" | "conf" | "env" | "xml" | "svg" => {
            text::extract(path)
        }

        // PDF
        "pdf" => pdf::extract(path),

        // DOCX
        "docx" => docx::extract(path),

        // CSV / TSV
        "csv" | "tsv" => csv_extractor::extract(path),

        // Imágenes (OCR vía Ollama Vision)
        "png" | "jpg" | "jpeg" | "bmp" | "tiff" | "webp" => {
            if let Some(ollama) = ollama {
                image::extract(path, ollama).await
            } else {
                // Sin Ollama, solo registrar metadatos básicos
                let mut content = ExtractedContent::empty();
                content
                    .extra
                    .insert("type".to_string(), "image".to_string());
                Ok(content)
            }
        }

        _ => Err(SoasError::UnsupportedFileType(extension)),
    }
}

/// Verifica si una extensión es soportada
pub fn is_supported_extension(ext: &str) -> bool {
    matches!(
        ext.to_lowercase().as_str(),
        "txt"
            | "md"
            | "markdown"
            | "rst"
            | "log"
            | "rs"
            | "py"
            | "js"
            | "ts"
            | "jsx"
            | "tsx"
            | "html"
            | "css"
            | "scss"
            | "go"
            | "java"
            | "c"
            | "cpp"
            | "h"
            | "hpp"
            | "cs"
            | "rb"
            | "php"
            | "json"
            | "yaml"
            | "yml"
            | "toml"
            | "ini"
            | "xml"
            | "svg"
            | "pdf"
            | "docx"
            | "csv"
            | "tsv"
            | "png"
            | "jpg"
            | "jpeg"
            | "bmp"
            | "tiff"
            | "webp"
    )
}
