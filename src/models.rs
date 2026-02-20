use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

// ─────────────────────────────────────────────
//  Archivo Indexado
// ─────────────────────────────────────────────

/// Representa un archivo que ha sido indexado por SOAS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedFile {
    /// Identificador único
    pub id: String,
    /// Ruta absoluta al archivo real
    pub path: PathBuf,
    /// Nombre del archivo
    pub filename: String,
    /// Extensión del archivo (sin punto)
    pub extension: String,
    /// Tipo MIME detectado
    pub mime_type: String,
    /// Tamaño en bytes
    pub size: u64,
    /// Hash SHA-256 del contenido
    pub content_hash: String,
    /// Vista previa del contenido extraído (primeros ~500 chars)
    pub content_preview: String,
    /// Contenido completo extraído (para embeddings)
    pub content_full: String,
    /// Metadatos adicionales (título, autor, etc.)
    pub metadata: FileMetadata,
    /// Fecha de creación del archivo
    pub created_at: DateTime<Utc>,
    /// Fecha de última modificación
    pub modified_at: DateTime<Utc>,
    /// Fecha en que fue indexado
    pub indexed_at: DateTime<Utc>,
    /// Estado del indexado
    pub index_status: IndexStatus,
}

impl IndexedFile {
    pub fn new(path: PathBuf) -> Self {
        let filename = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let extension = path
            .extension()
            .map(|e| e.to_string_lossy().to_string())
            .unwrap_or_default();

        Self {
            id: Uuid::new_v4().to_string(),
            path,
            filename,
            extension,
            mime_type: String::new(),
            size: 0,
            content_hash: String::new(),
            content_preview: String::new(),
            content_full: String::new(),
            metadata: FileMetadata::default(),
            created_at: Utc::now(),
            modified_at: Utc::now(),
            indexed_at: Utc::now(),
            index_status: IndexStatus::Pending,
        }
    }
}

/// Estado del indexado de un archivo
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IndexStatus {
    /// Pendiente de indexar
    Pending,
    /// Contenido extraído, falta embedding
    ContentExtracted,
    /// Completamente indexado con embedding
    Indexed,
    /// Error durante el indexado
    Failed(String),
    /// Archivo modificado, necesita re-indexar
    Stale,
}

impl IndexStatus {
    pub fn as_str(&self) -> &str {
        match self {
            IndexStatus::Pending => "pending",
            IndexStatus::ContentExtracted => "content_extracted",
            IndexStatus::Indexed => "indexed",
            IndexStatus::Failed(_) => "failed",
            IndexStatus::Stale => "stale",
        }
    }

    pub fn from_str_with_detail(s: &str, detail: Option<&str>) -> Self {
        match s {
            "pending" => IndexStatus::Pending,
            "content_extracted" => IndexStatus::ContentExtracted,
            "indexed" => IndexStatus::Indexed,
            "failed" => IndexStatus::Failed(detail.unwrap_or("unknown").to_string()),
            "stale" => IndexStatus::Stale,
            _ => IndexStatus::Pending,
        }
    }
}

// ─────────────────────────────────────────────
//  Metadatos de Archivo
// ─────────────────────────────────────────────

/// Metadatos extraídos del archivo
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FileMetadata {
    /// Título del documento (si está disponible)
    pub title: Option<String>,
    /// Autor del documento
    pub author: Option<String>,
    /// Descripción o resumen generado por LLM
    pub description: Option<String>,
    /// Palabras clave extraídas
    pub keywords: Vec<String>,
    /// Etiquetas semánticas generadas por LLM (temas, entidades, conceptos)
    #[serde(default)]
    pub semantic_tags: Vec<String>,
    /// Grupo de tipo de contenido: "documento", "imagen", "hoja_calculo", "codigo", "archivo"
    #[serde(default)]
    pub content_type_group: Option<String>,
    /// Número de páginas (para PDF/DOCX)
    pub page_count: Option<u32>,
    /// Idioma detectado
    pub language: Option<String>,
    /// Metadatos adicionales como pares clave-valor
    pub extra: std::collections::HashMap<String, String>,
}

// ─────────────────────────────────────────────
//  Organización Virtual
// ─────────────────────────────────────────────

/// Categoría virtual para organizar archivos
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Category {
    /// Identificador único
    pub id: String,
    /// Nombre visible de la categoría
    pub name: String,
    /// Descripción de la categoría
    pub description: String,
    /// ID de la categoría padre (para jerarquía)
    pub parent_id: Option<String>,
    /// Icono de la categoría (emoji o nombre de icono)
    pub icon: Option<String>,
    /// Color de la categoría (hex)
    pub color: Option<String>,
    /// Orden de display
    pub sort_order: i32,
    /// Reglas de auto-clasificación
    pub auto_rules: Vec<CategoryRule>,
    /// Fecha de creación
    pub created_at: DateTime<Utc>,
}

impl Category {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.into(),
            description: description.into(),
            parent_id: None,
            icon: None,
            color: None,
            sort_order: 0,
            auto_rules: vec![],
            created_at: Utc::now(),
        }
    }
}

/// Regla para auto-clasificar archivos en categorías
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryRule {
    /// Tipo de regla
    pub rule_type: RuleType,
    /// Patrón o valor de la regla
    pub pattern: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleType {
    /// Coincidencia por extensión de archivo
    Extension,
    /// Coincidencia por tipo MIME
    MimeType,
    /// Coincidencia por nombre de archivo (glob)
    FileName,
    /// Coincidencia por ruta (glob)
    Path,
    /// Coincidencia por contenido (búsqueda de texto)
    Content,
    /// Coincidencia semántica (por similitud de embedding)
    Semantic,
}

/// Archivo virtual: referencia visual sin mover el archivo real
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualFile {
    /// ID del archivo indexado
    pub file_id: String,
    /// Nombre virtual para mostrar en la UI
    pub virtual_name: String,
    /// ID de la categoría a la que pertenece
    pub category_id: String,
    /// Notas del usuario sobre el archivo
    pub notes: Option<String>,
    /// Tags personalizados
    pub tags: Vec<String>,
    /// Orden dentro de la categoría
    pub sort_order: i32,
    /// ¿Fue auto-clasificado o manual?
    pub auto_classified: bool,
}

// ─────────────────────────────────────────────
//  Búsqueda
// ─────────────────────────────────────────────

/// Consulta de búsqueda
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    /// Texto de la consulta del usuario
    pub text: String,
    /// Filtros opcionales
    pub filters: SearchFilters,
    /// Número máximo de resultados
    pub limit: usize,
    /// Umbral mínimo de relevancia (0.0 - 1.0)
    pub min_score: f32,
}

impl SearchQuery {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            filters: SearchFilters::default(),
            limit: 10,
            min_score: 0.40,
        }
    }
}

/// Filtros para búsqueda
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchFilters {
    /// Filtrar por extensiones
    pub extensions: Vec<String>,
    /// Filtrar por tipo MIME
    pub mime_types: Vec<String>,
    /// Filtrar por categoría
    pub category_ids: Vec<String>,
    /// Filtrar por rango de fechas
    pub date_from: Option<DateTime<Utc>>,
    pub date_to: Option<DateTime<Utc>>,
    /// Filtrar por tamaño mínimo (bytes)
    pub min_size: Option<u64>,
    /// Filtrar por tamaño máximo (bytes)
    pub max_size: Option<u64>,
    /// Filtrar por ruta (debe contener este string)
    pub path_contains: Option<String>,
}

/// Resultado de búsqueda
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Archivo encontrado
    pub file: IndexedFile,
    /// Puntuación de relevancia (0.0 - 1.0)
    pub score: f32,
    /// Información virtual si existe
    pub virtual_info: Option<VirtualFile>,
    /// Fragmento relevante del contenido
    pub snippet: String,
    /// Explicación de por qué es relevante
    pub explanation: Option<String>,
}

// ─────────────────────────────────────────────
//  Embeddings
// ─────────────────────────────────────────────

/// Un vector de embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// ID del archivo asociado
    pub file_id: String,
    /// Vector de embedding
    pub vector: Vec<f32>,
    /// Modelo usado para generar el embedding
    pub model: String,
    /// Fecha de generación
    pub created_at: DateTime<Utc>,
}

// ─────────────────────────────────────────────
//  Estadísticas e Indexado
// ─────────────────────────────────────────────

/// Carpeta que está siendo monitoreada para indexación
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchedFolder {
    /// Identificador único
    pub id: String,
    /// Ruta de la carpeta
    pub path: PathBuf,
    /// Nombre visible
    pub name: String,
    /// ¿Indexar recursivamente?
    pub recursive: bool,
    /// Patrones de exclusión (glob)
    pub exclude_patterns: Vec<String>,
    /// ¿Está activo el monitoreo?
    pub active: bool,
    /// Última vez que se escaneó
    pub last_scan: Option<DateTime<Utc>>,
    /// Fecha de creación
    pub created_at: DateTime<Utc>,
}

impl WatchedFolder {
    pub fn new(path: PathBuf, name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            path,
            name: name.into(),
            recursive: true,
            exclude_patterns: vec![
                ".git".to_string(),
                "node_modules".to_string(),
                ".cache".to_string(),
                "__pycache__".to_string(),
                "target".to_string(),
            ],
            active: true,
            last_scan: None,
            created_at: Utc::now(),
        }
    }
}

/// Estadísticas del sistema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStats {
    /// Total de archivos indexados
    pub total_files: u64,
    /// Total de archivos con embeddings
    pub total_embedded: u64,
    /// Total de categorías
    pub total_categories: u64,
    /// Total de carpetas monitoreadas
    pub total_watched_folders: u64,
    /// Tamaño total de la base de datos
    pub database_size_bytes: u64,
    /// Espacio en disco de archivos indexados
    pub indexed_files_size_bytes: u64,
}

/// Progreso de indexación
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexProgress {
    /// Archivos totales a procesar
    pub total: u64,
    /// Archivos procesados
    pub processed: u64,
    /// Archivos con error
    pub failed: u64,
    /// Archivo actual siendo procesado
    pub current_file: Option<String>,
    /// Fase actual
    pub phase: IndexPhase,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexPhase {
    Scanning,
    ExtractingContent,
    GeneratingEmbeddings,
    Classifying,
    Complete,
}
