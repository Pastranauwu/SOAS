use thiserror::Error;

/// Errores del sistema SOAS
#[derive(Error, Debug)]
pub enum SoasError {
    #[error("Error de base de datos: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Error de I/O: {0}")]
    Io(#[from] std::io::Error),

    #[error("Error de red: {0}")]
    Network(#[from] reqwest::Error),

    #[error("Error de serialización JSON: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Error de CSV: {0}")]
    Csv(#[from] csv::Error),

    #[error("Error de extracción de contenido: {0}")]
    ContentExtraction(String),

    #[error("Error de Ollama: {0}")]
    Ollama(String),

    #[error("Error de embeddings: {0}")]
    Embedding(String),

    #[error("Error del vector store: {0}")]
    VectorStore(String),

    #[error("Archivo no encontrado: {0}")]
    FileNotFound(String),

    #[error("Tipo de archivo no soportado: {0}")]
    UnsupportedFileType(String),

    #[error("Error de configuración: {0}")]
    Config(String),

    #[error("Error del indexador: {0}")]
    Indexer(String),

    #[error("Operación cancelada")]
    Cancelled,

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, SoasError>;
