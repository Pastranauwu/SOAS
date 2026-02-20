use crate::content::ExtractedContent;
use crate::error::Result;
use std::path::Path;
use tracing::debug;

/// Extrae contenido de archivos CSV/TSV
pub fn extract(path: &Path) -> Result<ExtractedContent> {
    debug!("Extrayendo contenido de CSV: {:?}", path);

    let extension = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    let delimiter = if extension == "tsv" { b'\t' } else { b',' };

    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .flexible(true)
        .has_headers(true)
        .from_path(path)?;

    let mut text = String::new();
    let mut row_count: u32 = 0;
    let mut extra = std::collections::HashMap::new();

    // Agregar headers
    if let Ok(headers) = reader.headers() {
        let header_line: Vec<&str> = headers.iter().collect();
        text.push_str("Columnas: ");
        text.push_str(&header_line.join(", "));
        text.push('\n');
        extra.insert("columns".to_string(), header_line.join(","));
    }

    text.push('\n');

    // Leer filas (limitado para no consumir demasiada memoria)
    for record in reader.records().take(1000) {
        if let Ok(record) = record {
            let values: Vec<&str> = record.iter().collect();
            text.push_str(&values.join(" | "));
            text.push('\n');
            row_count += 1;
        }
    }

    extra.insert("row_count".to_string(), row_count.to_string());
    extra.insert("type".to_string(), "spreadsheet".to_string());

    Ok(ExtractedContent {
        text,
        title: path.file_stem().map(|s| s.to_string_lossy().to_string()),
        author: None,
        page_count: None,
        extra,
    })
}
