use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuración global del sistema SOAS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoasConfig {
    /// Configuración de Ollama
    pub ollama: OllamaConfig,
    /// Configuración de almacenamiento
    pub storage: StorageConfig,
    /// Configuración del indexador
    pub indexer: IndexerConfig,
    /// Configuración de búsqueda
    pub search: SearchConfig,
}

impl Default for SoasConfig {
    fn default() -> Self {
        Self {
            ollama: OllamaConfig::default(),
            storage: StorageConfig::default(),
            indexer: IndexerConfig::default(),
            search: SearchConfig::default(),
        }
    }
}

impl SoasConfig {
    /// Carga la configuración desde un archivo JSON
    pub fn load(path: &std::path::Path) -> crate::error::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Guarda la configuración en un archivo JSON
    pub fn save(&self, path: &std::path::Path) -> crate::error::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Devuelve la ruta por defecto del directorio de datos de SOAS
    pub fn default_data_dir() -> PathBuf {
        let base = dirs_or_default();
        base.join("soas")
    }
}

fn dirs_or_default() -> PathBuf {
    // Intentar usar XDG en Linux, AppData en Windows
    if cfg!(target_os = "linux") {
        std::env::var("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
                PathBuf::from(home).join(".local/share")
            })
    } else if cfg!(target_os = "windows") {
        std::env::var("APPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("C:\\ProgramData"))
    } else {
        PathBuf::from("/tmp")
    }
}

/// Configuración del servidor Ollama
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    /// URL base del servidor Ollama
    pub base_url: String,
    /// Modelo para generar embeddings
    pub embedding_model: String,
    /// Modelo de chat para consultas y clasificación
    pub chat_model: String,
    /// Modelo de visión para OCR de documentos e imágenes con texto
    pub vision_model: String,
    /// Modelo de visión para describir fotos e ilustraciones (qwen3-vl:4b)
    pub description_model: String,
    /// Timeout en segundos para las peticiones
    pub timeout_secs: u64,
    /// Timeout específico para modelos de visión (OCR/descripción de imágenes).
    /// Más corto que el general para no bloquear ~5 min si el modelo no responde.
    pub vision_timeout_secs: u64,
    /// Dimensión del vector de embedding (depende del modelo)
    pub embedding_dimensions: usize,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            embedding_model: "nomic-embed-text".to_string(),
            // qwen2.5:3b: mejor que llama3.2 siguiendo JSON estricto, mismo tamaño (~1.9GB)
            chat_model: "qwen2.5:3b".to_string(),
            // glm-ocr: DESHABILITADO — falla consistentemente con error de red
            // en la API chat de Ollama. Se mantiene la config por si se arregla.
            vision_model: "glm-ocr".to_string(),
            // moondream: VLM ultraligero (1.7GB) optimizado para CPU.
            // Visual encoder más rápido que qwen3-vl:2b y no requiere GPU.
            // qwen3-vl:2b fallaba consistentemente en CPU: timeout o vacío.
            description_model: "moondream".to_string(),
            // 300s: inferencia en CPU para documentos largos
            timeout_secs: 300,
            // 90s: con archivos agrupados (imágenes al final), el primer request
            // paga la carga del modelo (~10-15s) + inferencia visual (~30-60s en CPU).
            // Requests posteriores son más rápidos porque el modelo ya está en RAM.
            vision_timeout_secs: 90,
            embedding_dimensions: 768, // nomic-embed-text default
        }
    }
}

/// Configuración de almacenamiento
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Directorio donde se guardan los datos de SOAS
    pub data_dir: PathBuf,
    /// Nombre del archivo de la base de datos SQLite
    pub db_filename: String,
    /// Nombre del archivo del vector store
    pub vector_store_filename: String,
}

impl Default for StorageConfig {
    fn default() -> Self {
        let data_dir = SoasConfig::default_data_dir();
        Self {
            data_dir,
            db_filename: "soas.db".to_string(),
            vector_store_filename: "vectors.bin".to_string(),
        }
    }
}

impl StorageConfig {
    /// Ruta completa a la base de datos SQLite
    pub fn db_path(&self) -> PathBuf {
        self.data_dir.join(&self.db_filename)
    }

    /// Ruta completa al vector store
    pub fn vector_store_path(&self) -> PathBuf {
        self.data_dir.join(&self.vector_store_filename)
    }
}

/// Configuración del indexador
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexerConfig {
    /// Tamaño máximo de archivo a indexar (en bytes, default 100MB)
    pub max_file_size: u64,
    /// Extensiones de archivo a indexar (vacío = todas las soportadas)
    pub allowed_extensions: Vec<String>,
    /// Patrones glob a excluir
    pub exclude_patterns: Vec<String>,
    /// Número de archivos a procesar en paralelo
    pub batch_size: usize,
    /// Longitud máxima de contenido a extraer (en caracteres)
    pub max_content_length: usize,
    /// Longitud de la vista previa del contenido
    pub preview_length: usize,
    /// ¿Habilitar monitoreo de cambios en tiempo real?
    pub watch_enabled: bool,
    /// Intervalo en segundos para re-escaneo completo
    pub rescan_interval_secs: u64,
    /// ¿Usar LLM para enriquecer metadatos durante indexación?
    /// Genera descripción, keywords semánticos y etiquetas con describe_file.
    /// Más lento (~3-5s extra por archivo) pero mejora mucho la búsqueda.
    pub use_llm_enrichment: bool,
}

impl Default for IndexerConfig {
    fn default() -> Self {
        Self {
            max_file_size: 100 * 1024 * 1024, // 100MB
            allowed_extensions: vec![
                // Documentos
                "pdf".to_string(),
                "docx".to_string(),
                "doc".to_string(),
                "txt".to_string(),
                "md".to_string(),
                "rtf".to_string(),
                // Hojas de cálculo
                "csv".to_string(),
                "tsv".to_string(),
                // Imágenes (para OCR)
                "png".to_string(),
                "jpg".to_string(),
                "jpeg".to_string(),
                "bmp".to_string(),
                "tiff".to_string(),
                "webp".to_string(),
                // Código
                "rs".to_string(),
                "py".to_string(),
                "js".to_string(),
                "ts".to_string(),
                "html".to_string(),
                "css".to_string(),
                "json".to_string(),
                "xml".to_string(),
                "yaml".to_string(),
                "yml".to_string(),
                "toml".to_string(),
            ],
            exclude_patterns: vec![
                "**/.git/**".to_string(),
                "**/node_modules/**".to_string(),
                "**/target/**".to_string(),
                "**/.cache/**".to_string(),
                "**/__pycache__/**".to_string(),
                "**/build/**".to_string(),
                "**/.DS_Store".to_string(),
                "**/Thumbs.db".to_string(),
            ],
            batch_size: 10,
            max_content_length: 50_000,
            preview_length: 500,
            watch_enabled: true,
            rescan_interval_secs: 3600, // 1 hora
            use_llm_enrichment: true,
        }
    }
}

/// Configuración de búsqueda
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Número máximo de resultados por defecto
    pub default_limit: usize,
    /// Umbral mínimo de similitud por defecto (0.0 - 1.0)
    pub default_min_score: f32,
    /// ¿Usar LLM para mejorar la consulta?
    pub use_query_enhancement: bool,
    /// ¿Generar explicaciones de relevancia?
    pub generate_explanations: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            default_limit: 10,
            default_min_score: 0.40,
            use_query_enhancement: true,
            generate_explanations: false,
        }
    }
}
