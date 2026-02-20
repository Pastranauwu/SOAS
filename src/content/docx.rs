use crate::content::ExtractedContent;
use crate::error::{Result, SoasError};
use quick_xml::events::Event;
use quick_xml::reader::Reader;
use std::io::Read;
use std::path::Path;
use tracing::{debug, warn};

/// Extrae texto de archivos DOCX
///
/// Un archivo DOCX es un ZIP que contiene XML.
/// El contenido principal está en word/document.xml
pub fn extract(path: &Path) -> Result<ExtractedContent> {
    debug!("Extrayendo texto de DOCX: {:?}", path);

    let file = std::fs::File::open(path)?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| {
        SoasError::ContentExtraction(format!("No se pudo abrir DOCX como ZIP: {}", e))
    })?;

    let mut text = String::new();
    let mut title = None;
    let mut author = None;

    // Extraer contenido principal de word/document.xml
    if let Ok(mut doc_entry) = archive.by_name("word/document.xml") {
        let mut xml_content = String::new();
        doc_entry.read_to_string(&mut xml_content)?;
        text = extract_text_from_docx_xml(&xml_content);
    }

    // Intentar extraer metadatos de docProps/core.xml
    if let Ok(mut props_entry) = archive.by_name("docProps/core.xml") {
        let mut xml_content = String::new();
        props_entry.read_to_string(&mut xml_content)?;
        let (t, a) = extract_metadata_from_core_xml(&xml_content);
        title = t;
        author = a;
    }

    // Contar páginas desde docProps/app.xml si existe
    let page_count = if let Ok(mut app_entry) = archive.by_name("docProps/app.xml") {
        let mut xml_content = String::new();
        app_entry.read_to_string(&mut xml_content)?;
        extract_page_count(&xml_content)
    } else {
        None
    };

    Ok(ExtractedContent {
        text,
        title,
        author,
        page_count,
        extra: std::collections::HashMap::new(),
    })
}

/// Extrae texto plano del XML del documento Word
fn extract_text_from_docx_xml(xml: &str) -> String {
    let mut reader = Reader::from_str(xml);
    let mut text = String::new();
    let mut in_text_element = false;
    let mut in_paragraph = false;

    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) => {
                let local_name = e.local_name();
                let name = std::str::from_utf8(local_name.as_ref()).unwrap_or("");
                match name {
                    "t" => in_text_element = true,
                    "p" => in_paragraph = true,
                    _ => {}
                }
            }
            Ok(Event::End(ref e)) => {
                let local_name = e.local_name();
                let name = std::str::from_utf8(local_name.as_ref()).unwrap_or("");
                match name {
                    "t" => in_text_element = false,
                    "p" => {
                        if in_paragraph {
                            text.push('\n');
                            in_paragraph = false;
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(e)) => {
                if in_text_element {
                    if let Ok(t) = e.unescape() {
                        text.push_str(&t);
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                warn!("Error parseando DOCX XML: {}", e);
                break;
            }
            _ => {}
        }
    }

    text.trim().to_string()
}

/// Extrae título y autor de core.xml
fn extract_metadata_from_core_xml(xml: &str) -> (Option<String>, Option<String>) {
    let mut reader = Reader::from_str(xml);
    let mut title = None;
    let mut author = None;
    let mut current_element = String::new();

    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) => {
                let local_name = e.local_name();
                current_element =
                    std::str::from_utf8(local_name.as_ref()).unwrap_or("").to_string();
            }
            Ok(Event::Text(e)) => {
                if let Ok(t) = e.unescape() {
                    let text = t.trim().to_string();
                    if !text.is_empty() {
                        match current_element.as_str() {
                            "title" => title = Some(text),
                            "creator" => author = Some(text),
                            _ => {}
                        }
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
    }

    (title, author)
}

/// Extrae el conteo de páginas de app.xml
fn extract_page_count(xml: &str) -> Option<u32> {
    let mut reader = Reader::from_str(xml);
    let mut current_element = String::new();

    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) => {
                let local_name = e.local_name();
                current_element =
                    std::str::from_utf8(local_name.as_ref()).unwrap_or("").to_string();
            }
            Ok(Event::Text(e)) => {
                if current_element == "Pages" {
                    if let Ok(t) = e.unescape() {
                        return t.trim().parse().ok();
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
    }

    None
}
