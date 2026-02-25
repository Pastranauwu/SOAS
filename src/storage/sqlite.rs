use crate::error::Result;
use crate::models::*;
use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;
use tracing::{debug, info};

/// Capa de almacenamiento basada en SQLite
pub struct SqliteStorage {
    conn: Connection,
}

impl SqliteStorage {
    /// Crea o abre una base de datos SQLite en la ruta indicada
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(path)?;

        // Configuración de rendimiento
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA cache_size = -64000;
             PRAGMA temp_store = MEMORY;
             PRAGMA mmap_size = 268435456;
               PRAGMA wal_autocheckpoint = 1000;
               PRAGMA journal_size_limit = 67108864;
               PRAGMA busy_timeout = 5000;
             PRAGMA foreign_keys = ON;",
        )?;

        let storage = Self { conn };
        storage.initialize_schema()?;

        info!("Base de datos SQLite abierta en: {:?}", path);
        Ok(storage)
    }

    /// Crea una base de datos en memoria (para tests)
    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        let storage = Self { conn };
        storage.initialize_schema()?;
        Ok(storage)
    }

    /// Inicializa el esquema de la base de datos
    fn initialize_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            "
            -- Carpetas monitoreadas
            CREATE TABLE IF NOT EXISTS watched_folders (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                recursive INTEGER NOT NULL DEFAULT 1,
                exclude_patterns TEXT NOT NULL DEFAULT '[]',
                active INTEGER NOT NULL DEFAULT 1,
                last_scan TEXT,
                created_at TEXT NOT NULL
            );

            -- Archivos indexados
            CREATE TABLE IF NOT EXISTS indexed_files (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                extension TEXT NOT NULL DEFAULT '',
                mime_type TEXT NOT NULL DEFAULT '',
                size INTEGER NOT NULL DEFAULT 0,
                content_hash TEXT NOT NULL DEFAULT '',
                content_preview TEXT NOT NULL DEFAULT '',
                content_full TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                modified_at TEXT NOT NULL,
                indexed_at TEXT NOT NULL,
                index_status TEXT NOT NULL DEFAULT 'pending',
                index_status_detail TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_files_path ON indexed_files(path);
            CREATE INDEX IF NOT EXISTS idx_files_extension ON indexed_files(extension);
            CREATE INDEX IF NOT EXISTS idx_files_status ON indexed_files(index_status);
            CREATE INDEX IF NOT EXISTS idx_files_hash ON indexed_files(content_hash);
            CREATE INDEX IF NOT EXISTS idx_files_filename ON indexed_files(filename COLLATE NOCASE);

            -- Categorías virtuales
            CREATE TABLE IF NOT EXISTS categories (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                parent_id TEXT,
                icon TEXT,
                color TEXT,
                sort_order INTEGER NOT NULL DEFAULT 0,
                auto_rules TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                FOREIGN KEY (parent_id) REFERENCES categories(id) ON DELETE SET NULL
            );

            -- Archivos virtuales (organización sin mover archivos reales)
            CREATE TABLE IF NOT EXISTS virtual_files (
                file_id TEXT NOT NULL,
                category_id TEXT NOT NULL,
                virtual_name TEXT NOT NULL,
                notes TEXT,
                tags TEXT NOT NULL DEFAULT '[]',
                sort_order INTEGER NOT NULL DEFAULT 0,
                auto_classified INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (file_id, category_id),
                FOREIGN KEY (file_id) REFERENCES indexed_files(id) ON DELETE CASCADE,
                FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_virtual_category ON virtual_files(category_id);

            -- FTS5 para búsqueda full-text
            CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
                filename,
                content_preview,
                content_full,
                content='indexed_files',
                content_rowid='rowid'
            );

            -- Triggers para mantener FTS sincronizado
            CREATE TRIGGER IF NOT EXISTS files_fts_insert AFTER INSERT ON indexed_files BEGIN
                INSERT INTO files_fts(rowid, filename, content_preview, content_full)
                VALUES (NEW.rowid, NEW.filename, NEW.content_preview, NEW.content_full);
            END;

            CREATE TRIGGER IF NOT EXISTS files_fts_delete AFTER DELETE ON indexed_files BEGIN
                INSERT INTO files_fts(files_fts, rowid, filename, content_preview, content_full)
                VALUES ('delete', OLD.rowid, OLD.filename, OLD.content_preview, OLD.content_full);
            END;

            CREATE TRIGGER IF NOT EXISTS files_fts_update AFTER UPDATE ON indexed_files BEGIN
                INSERT INTO files_fts(files_fts, rowid, filename, content_preview, content_full)
                VALUES ('delete', OLD.rowid, OLD.filename, OLD.content_preview, OLD.content_full);
                INSERT INTO files_fts(rowid, filename, content_preview, content_full)
                VALUES (NEW.rowid, NEW.filename, NEW.content_preview, NEW.content_full);
            END;
            ",
        )?;

        debug!("Esquema de base de datos inicializado");
        Ok(())
    }

    // ─────────────────────────────────────────
    //  Watched Folders
    // ─────────────────────────────────────────

    /// Agrega una carpeta al monitoreo
    pub fn add_watched_folder(&self, folder: &WatchedFolder) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO watched_folders
             (id, path, name, recursive, exclude_patterns, active, last_scan, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                folder.id,
                folder.path.to_string_lossy().to_string(),
                folder.name,
                folder.recursive as i32,
                serde_json::to_string(&folder.exclude_patterns)?,
                folder.active as i32,
                folder.last_scan.map(|d| d.to_rfc3339()),
                folder.created_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    /// Obtiene todas las carpetas monitoreadas
    pub fn get_watched_folders(&self) -> Result<Vec<WatchedFolder>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path, name, recursive, exclude_patterns, active, last_scan, created_at
             FROM watched_folders ORDER BY name",
        )?;

        let folders = stmt
            .query_map([], |row| {
                let exclude_json: String = row.get(4)?;
                let last_scan_str: Option<String> = row.get(6)?;
                let created_str: String = row.get(7)?;

                Ok(WatchedFolder {
                    id: row.get(0)?,
                    path: std::path::PathBuf::from(row.get::<_, String>(1)?),
                    name: row.get(2)?,
                    recursive: row.get::<_, i32>(3)? != 0,
                    exclude_patterns: serde_json::from_str(&exclude_json).unwrap_or_default(),
                    active: row.get::<_, i32>(5)? != 0,
                    last_scan: last_scan_str.and_then(|s| {
                        chrono::DateTime::parse_from_rfc3339(&s)
                            .ok()
                            .map(|d| d.with_timezone(&Utc))
                    }),
                    created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
                        .map(|d| d.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(folders)
    }

    /// Elimina una carpeta del monitoreo
    pub fn remove_watched_folder(&self, folder_id: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM watched_folders WHERE id = ?1",
            params![folder_id],
        )?;
        Ok(())
    }

    /// Actualiza la fecha del último escaneo
    pub fn update_folder_last_scan(&self, folder_id: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE watched_folders SET last_scan = ?1 WHERE id = ?2",
            params![Utc::now().to_rfc3339(), folder_id],
        )?;
        Ok(())
    }

    // ─────────────────────────────────────────
    //  Indexed Files
    // ─────────────────────────────────────────

    /// Inserta o actualiza un archivo indexado
    pub fn upsert_file(&self, file: &IndexedFile) -> Result<()> {
        let status_detail = match &file.index_status {
            IndexStatus::Failed(detail) => Some(detail.clone()),
            _ => None,
        };

        self.conn.execute(
            "INSERT OR REPLACE INTO indexed_files
             (id, path, filename, extension, mime_type, size,
              content_hash, content_preview, content_full,
              metadata_json, created_at, modified_at, indexed_at,
              index_status, index_status_detail)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            params![
                file.id,
                file.path.to_string_lossy().to_string(),
                file.filename,
                file.extension,
                file.mime_type,
                file.size as i64,
                file.content_hash,
                file.content_preview,
                file.content_full,
                serde_json::to_string(&file.metadata)?,
                file.created_at.to_rfc3339(),
                file.modified_at.to_rfc3339(),
                file.indexed_at.to_rfc3339(),
                file.index_status.as_str(),
                status_detail,
            ],
        )?;
        Ok(())
    }

    /// Obtiene un archivo por su ID
    pub fn get_file_by_id(&self, file_id: &str) -> Result<Option<IndexedFile>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path, filename, extension, mime_type, size,
                    content_hash, content_preview, content_full,
                    metadata_json, created_at, modified_at, indexed_at,
                    index_status, index_status_detail
             FROM indexed_files WHERE id = ?1",
        )?;

        let file = stmt
            .query_row(params![file_id], |row| self.row_to_indexed_file(row))
            .optional()?;

        Ok(file)
    }

    /// Obtiene un archivo por su ruta
    pub fn get_file_by_path(&self, path: &Path) -> Result<Option<IndexedFile>> {
        let path_str = path.to_string_lossy().to_string();
        let mut stmt = self.conn.prepare(
            "SELECT id, path, filename, extension, mime_type, size,
                    content_hash, content_preview, content_full,
                    metadata_json, created_at, modified_at, indexed_at,
                    index_status, index_status_detail
             FROM indexed_files WHERE path = ?1",
        )?;

        let file = stmt
            .query_row(params![path_str], |row| self.row_to_indexed_file(row))
            .optional()?;

        Ok(file)
    }

    /// Lista archivos por estado
    pub fn get_files_by_status(&self, status: &str, limit: usize) -> Result<Vec<IndexedFile>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path, filename, extension, mime_type, size,
                    content_hash, content_preview, content_full,
                    metadata_json, created_at, modified_at, indexed_at,
                    index_status, index_status_detail
             FROM indexed_files WHERE index_status = ?1 LIMIT ?2",
        )?;

        let files = stmt
            .query_map(params![status, limit as i64], |row| {
                self.row_to_indexed_file(row)
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(files)
    }

    /// Busca archivos por texto (FTS5)
    pub fn search_fts(&self, query: &str, limit: usize) -> Result<Vec<(IndexedFile, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT f.id, f.path, f.filename, f.extension, f.mime_type, f.size,
                    f.content_hash, f.content_preview, f.content_full,
                    f.metadata_json, f.created_at, f.modified_at, f.indexed_at,
                    f.index_status, f.index_status_detail,
                    rank
             FROM indexed_files f
             JOIN files_fts fts ON f.rowid = fts.rowid
             WHERE files_fts MATCH ?1
             ORDER BY rank
             LIMIT ?2",
        )?;

        let results = stmt
            .query_map(params![query, limit as i64], |row| {
                let file = self.row_to_indexed_file(row)?;
                let rank: f64 = row.get(15)?;
                Ok((file, rank))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(results)
    }

    /// Elimina un archivo por ID
    pub fn delete_file(&self, file_id: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM indexed_files WHERE id = ?1",
            params![file_id],
        )?;
        Ok(())
    }

    /// Elimina archivos cuya ruta comience con un prefijo (al quitar una carpeta)
    pub fn delete_files_by_path_prefix(&self, prefix: &str) -> Result<u64> {
        let count = self.conn.execute(
            "DELETE FROM indexed_files WHERE path LIKE ?1",
            params![format!("{}%", prefix)],
        )?;
        Ok(count as u64)
    }

    /// Actualiza solo el estado de indexación de un archivo
    pub fn update_file_status(&self, file_id: &str, status: &IndexStatus) -> Result<()> {
        let detail = match status {
            IndexStatus::Failed(d) => Some(d.clone()),
            _ => None,
        };
        self.conn.execute(
            "UPDATE indexed_files SET index_status = ?1, index_status_detail = ?2 WHERE id = ?3",
            params![status.as_str(), detail, file_id],
        )?;
        Ok(())
    }

    /// Número total de archivos indexados
    pub fn count_files(&self) -> Result<u64> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM indexed_files", [], |row| row.get(0))?;
        Ok(count as u64)
    }

    /// Número de archivos por estado
    pub fn count_files_by_status(&self, status: &str) -> Result<u64> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM indexed_files WHERE index_status = ?1",
            params![status],
            |row| row.get(0),
        )?;
        Ok(count as u64)
    }

    /// Resetea el estado de todos los archivos indexados a 'content_extracted'
    /// para que se les regeneren los embeddings sin re-extraer contenido
    pub fn reset_all_to_content_extracted(&self) -> Result<u64> {
        let count = self.conn.execute(
            "UPDATE indexed_files SET index_status = 'content_extracted', index_status_detail = NULL
             WHERE index_status = 'indexed'",
            [],
        )?;
        Ok(count as u64)
    }

    /// Resetea el estado de todos los archivos a 'pending' para forzar
    /// re-extracción de contenido + re-generación de embeddings
    pub fn reset_all_to_pending(&self) -> Result<u64> {
        let count = self.conn.execute(
            "UPDATE indexed_files SET index_status = 'pending', index_status_detail = NULL,
             content_hash = ''",
            [],
        )?;
        Ok(count as u64)
    }

    /// Resetea solo las imágenes a 'pending' para re-extraer con nuevo prompt de visión
    pub fn reset_images_to_pending(&self) -> Result<u64> {
        let count = self.conn.execute(
            "UPDATE indexed_files SET index_status = 'pending', index_status_detail = NULL,
             content_hash = ''
             WHERE extension IN ('jpg','jpeg','png','bmp','tiff','webp','gif','svg','heic')",
            [],
        )?;
        Ok(count as u64)
    }

    /// Obtiene todos los archivos indexados (para rebuild)
    pub fn get_all_indexed_files(&self) -> Result<Vec<IndexedFile>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path, filename, extension, mime_type, size,
                    content_hash, content_preview, content_full,
                    metadata_json, created_at, modified_at, indexed_at,
                    index_status, index_status_detail
             FROM indexed_files
             WHERE content_full IS NOT NULL AND content_full != ''",
        )?;

        let results = stmt
            .query_map([], |row| self.row_to_indexed_file(row))?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(results)
    }

    /// Obtiene todos los IDs de archivos con sus hashes (para detección de cambios)
    pub fn get_all_file_paths_and_hashes(&self) -> Result<Vec<(String, String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path, content_hash FROM indexed_files",
        )?;

        let results = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(results)
    }

    // ─────────────────────────────────────────
    //  Categories
    // ─────────────────────────────────────────

    /// Crea una categoría
    pub fn create_category(&self, category: &Category) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO categories
             (id, name, description, parent_id, icon, color, sort_order, auto_rules, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                category.id,
                category.name,
                category.description,
                category.parent_id,
                category.icon,
                category.color,
                category.sort_order,
                serde_json::to_string(&category.auto_rules)?,
                category.created_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    /// Obtiene todas las categorías
    pub fn get_categories(&self) -> Result<Vec<Category>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, description, parent_id, icon, color, sort_order, auto_rules, created_at
             FROM categories ORDER BY sort_order, name",
        )?;

        let categories = stmt
            .query_map([], |row| {
                let rules_json: String = row.get(7)?;
                let created_str: String = row.get(8)?;

                Ok(Category {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    description: row.get(2)?,
                    parent_id: row.get(3)?,
                    icon: row.get(4)?,
                    color: row.get(5)?,
                    sort_order: row.get(6)?,
                    auto_rules: serde_json::from_str(&rules_json).unwrap_or_default(),
                    created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
                        .map(|d| d.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(categories)
    }

    /// Elimina una categoría
    pub fn delete_category(&self, category_id: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM categories WHERE id = ?1",
            params![category_id],
        )?;
        Ok(())
    }

    // ─────────────────────────────────────────
    //  Virtual Files
    // ─────────────────────────────────────────

    /// Asigna un archivo a una categoría con nombre virtual
    pub fn set_virtual_file(&self, vf: &VirtualFile) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO virtual_files
             (file_id, category_id, virtual_name, notes, tags, sort_order, auto_classified)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                vf.file_id,
                vf.category_id,
                vf.virtual_name,
                vf.notes,
                serde_json::to_string(&vf.tags)?,
                vf.sort_order,
                vf.auto_classified as i32,
            ],
        )?;
        Ok(())
    }

    /// Obtiene archivos virtuales de una categoría
    pub fn get_virtual_files_by_category(
        &self,
        category_id: &str,
    ) -> Result<Vec<(VirtualFile, IndexedFile)>> {
        let mut stmt = self.conn.prepare(
            "SELECT vf.file_id, vf.category_id, vf.virtual_name, vf.notes, vf.tags,
                    vf.sort_order, vf.auto_classified,
                    f.id, f.path, f.filename, f.extension, f.mime_type, f.size,
                    f.content_hash, f.content_preview, f.content_full,
                    f.metadata_json, f.created_at, f.modified_at, f.indexed_at,
                    f.index_status, f.index_status_detail
             FROM virtual_files vf
             JOIN indexed_files f ON f.id = vf.file_id
             WHERE vf.category_id = ?1
             ORDER BY vf.sort_order, vf.virtual_name",
        )?;

        let results = stmt
            .query_map(params![category_id], |row| {
                let tags_json: String = row.get(4)?;
                let vf = VirtualFile {
                    file_id: row.get(0)?,
                    category_id: row.get(1)?,
                    virtual_name: row.get(2)?,
                    notes: row.get(3)?,
                    tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                    sort_order: row.get(5)?,
                    auto_classified: row.get::<_, i32>(6)? != 0,
                };

                let metadata_json: String = row.get(16)?;
                let created_str: String = row.get(17)?;
                let modified_str: String = row.get(18)?;
                let indexed_str: String = row.get(19)?;
                let status_str: String = row.get(20)?;
                let status_detail: Option<String> = row.get(21)?;

                let file = IndexedFile {
                    id: row.get(7)?,
                    path: std::path::PathBuf::from(row.get::<_, String>(8)?),
                    filename: row.get(9)?,
                    extension: row.get(10)?,
                    mime_type: row.get(11)?,
                    size: row.get::<_, i64>(12)? as u64,
                    content_hash: row.get(13)?,
                    content_preview: row.get(14)?,
                    content_full: row.get(15)?,
                    metadata: serde_json::from_str(&metadata_json).unwrap_or_default(),
                    created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
                        .map(|d| d.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    modified_at: chrono::DateTime::parse_from_rfc3339(&modified_str)
                        .map(|d| d.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    indexed_at: chrono::DateTime::parse_from_rfc3339(&indexed_str)
                        .map(|d| d.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    index_status: IndexStatus::from_str_with_detail(
                        &status_str,
                        status_detail.as_deref(),
                    ),
                };

                Ok((vf, file))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(results)
    }

    /// Elimina la asignación virtual de un archivo en una categoría
    pub fn remove_virtual_file(&self, file_id: &str, category_id: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM virtual_files WHERE file_id = ?1 AND category_id = ?2",
            params![file_id, category_id],
        )?;
        Ok(())
    }

    /// Obtiene todas las asignaciones virtuales de un archivo
    pub fn get_virtual_files_for_file(&self, file_id: &str) -> Result<Vec<VirtualFile>> {
        let mut stmt = self.conn.prepare(
            "SELECT file_id, category_id, virtual_name, notes, tags, sort_order, auto_classified
             FROM virtual_files WHERE file_id = ?1",
        )?;

        let results = stmt
            .query_map(params![file_id], |row| {
                let tags_json: String = row.get(4)?;
                Ok(VirtualFile {
                    file_id: row.get(0)?,
                    category_id: row.get(1)?,
                    virtual_name: row.get(2)?,
                    notes: row.get(3)?,
                    tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                    sort_order: row.get(5)?,
                    auto_classified: row.get::<_, i32>(6)? != 0,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(results)
    }

    // ─────────────────────────────────────────
    //  Consultas del Gestor de Archivos
    // ─────────────────────────────────────────

    /// Búsqueda rápida tipo Spotlight en nombres de archivo e índice.
    ///
    /// Estrategia de ranking (igual que Spotlight):
    ///   1 → nombre exacto (sin extensión)
    ///   2 → nombre empieza por la consulta
    ///   3 → alguna palabra del nombre empieza por la consulta
    ///   4 → nombre contiene la consulta en cualquier posición
    ///   5 → la ruta completa contiene la consulta (coincidencia de carpeta)
    ///
    /// Retorna hasta `limit` resultados ordenados por ranking + tamaño decreciente.
    pub fn spotlight_search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(IndexedFile, u8)>> {
        if query.trim().is_empty() {
            return Ok(vec![]);
        }

        // Patrones para los distintos niveles de coincidencia
        let q_lower      = query.to_lowercase();
        let exact_pat    = q_lower.clone();          // nombre sin ext == query
        let starts_pat   = format!("{}%", q_lower);  // filename LIKE 'q%'
        let word_pat     = format!("% {}%", q_lower); // ' q' en algún lugar del nombre
        let contains_pat = format!("%{}%", q_lower);  // nombre contiene q
        let path_pat     = format!("%{}%", q_lower);  // ruta contiene q

        let mut stmt = self.conn.prepare(
            "SELECT
                f.id, f.path, f.filename, f.extension, f.mime_type, f.size,
                f.content_hash, f.content_preview, f.content_full,
                f.metadata_json, f.created_at, f.modified_at, f.indexed_at,
                f.index_status, f.index_status_detail,
                CASE
                    WHEN LOWER(REPLACE(f.filename, '.' || f.extension, '')) = ?1  THEN 1
                    WHEN LOWER(f.filename) LIKE ?2                                 THEN 2
                    WHEN LOWER(f.filename) LIKE ?3                                 THEN 3
                    WHEN LOWER(f.filename) LIKE ?4                                 THEN 4
                    WHEN LOWER(f.path)     LIKE ?5                                 THEN 5
                    ELSE 99
                END AS rank_score
             FROM indexed_files f
             WHERE LOWER(f.filename) LIKE ?4
                OR LOWER(f.path)     LIKE ?5
             ORDER BY rank_score ASC, f.size DESC
             LIMIT ?6",
        )?;

        let results = stmt
            .query_map(
                rusqlite::params![
                    exact_pat,
                    starts_pat,
                    word_pat,
                    contains_pat,
                    path_pat,
                    limit as i64,
                ],
                |row| {
                    let file  = self.row_to_indexed_file(row)?;
                    let score = row.get::<_, u8>(15)?;
                    Ok((file, score))
                },
            )?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(results)
    }

    /// Lista archivos indexados con paginación y filtros opcionales
    pub fn get_files_paginated(
        &self,
        limit: usize,
        offset: usize,
        extension: Option<&str>,
        path_prefix: Option<&str>,
        sort_by: Option<&str>,
    ) -> Result<Vec<IndexedFile>> {
        let order = match sort_by.unwrap_or("modified_at") {
            "name" => "filename ASC",
            "size" => "size DESC",
            "created_at" => "created_at DESC",
            "indexed_at" => "indexed_at DESC",
            _ => "modified_at DESC",
        };

        let sql = format!(
            "SELECT id, path, filename, extension, mime_type, size,
                    content_hash, content_preview, content_full,
                    metadata_json, created_at, modified_at, indexed_at,
                    index_status, index_status_detail
             FROM indexed_files
             WHERE (?1 IS NULL OR extension = ?1)
               AND (?2 IS NULL OR path LIKE ?2)
             ORDER BY {order}
             LIMIT ?3 OFFSET ?4"
        );

        let prefix_pattern = path_prefix.map(|p| format!("{}%", p));
        let mut stmt = self.conn.prepare(&sql)?;
        let files = stmt
            .query_map(
                rusqlite::params![
                    extension,
                    prefix_pattern,
                    limit as i64,
                    offset as i64
                ],
                |row| self.row_to_indexed_file(row),
            )?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(files)
    }

    /// Archivos modificados más recientemente
    pub fn get_recent_files(&self, limit: usize) -> Result<Vec<IndexedFile>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path, filename, extension, mime_type, size,
                    content_hash, content_preview, content_full,
                    metadata_json, created_at, modified_at, indexed_at,
                    index_status, index_status_detail
             FROM indexed_files
             ORDER BY modified_at DESC
             LIMIT ?1",
        )?;

        let files = stmt
            .query_map(params![limit as i64], |row| self.row_to_indexed_file(row))?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(files)
    }

    /// Archivos con alguna de las extensiones dadas
    pub fn get_files_by_extensions(
        &self,
        extensions: &[&str],
        limit: usize,
        offset: usize,
    ) -> Result<Vec<IndexedFile>> {
        if extensions.is_empty() {
            return Ok(vec![]);
        }
        let placeholders: Vec<String> = (1..=extensions.len()).map(|i| format!("?{}", i)).collect();
        let sql = format!(
            "SELECT id, path, filename, extension, mime_type, size,
                    content_hash, content_preview, content_full,
                    metadata_json, created_at, modified_at, indexed_at,
                    index_status, index_status_detail
             FROM indexed_files
             WHERE extension IN ({})
             ORDER BY modified_at DESC
             LIMIT ?{} OFFSET ?{}",
            placeholders.join(", "),
            extensions.len() + 1,
            extensions.len() + 2,
        );

        let mut stmt = self.conn.prepare(&sql)?;
        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = extensions
            .iter()
            .map(|e| Box::new(e.to_string()) as Box<dyn rusqlite::ToSql>)
            .collect();
        params_vec.push(Box::new(limit as i64));
        params_vec.push(Box::new(offset as i64));

        let params_ref: Vec<&dyn rusqlite::ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();
        let files = stmt
            .query_map(params_ref.as_slice(), |row| self.row_to_indexed_file(row))?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(files)
    }

    /// Extensiones únicas presentes en la base de datos (para filtros)
    pub fn get_distinct_extensions(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT extension FROM indexed_files
             WHERE extension != ''
             ORDER BY extension",
        )?;
        let exts = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(exts)
    }

    /// Total de archivos según filtros (para paginación)
    pub fn count_files_filtered(
        &self,
        extension: Option<&str>,
        path_prefix: Option<&str>,
    ) -> Result<u64> {
        let prefix_pattern = path_prefix.map(|p| format!("{}%", p));
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM indexed_files
             WHERE (?1 IS NULL OR extension = ?1)
               AND (?2 IS NULL OR path LIKE ?2)",
            rusqlite::params![extension, prefix_pattern],
            |row| row.get(0),
        )?;
        Ok(count as u64)
    }

    /// Actualiza los campos editables de una carpeta monitoreada
    pub fn update_watched_folder(&self, folder: &WatchedFolder) -> Result<()> {
        self.conn.execute(
            "UPDATE watched_folders
             SET name = ?1, recursive = ?2, exclude_patterns = ?3, active = ?4
             WHERE id = ?5",
            params![
                folder.name,
                folder.recursive as i32,
                serde_json::to_string(&folder.exclude_patterns)?,
                folder.active as i32,
                folder.id,
            ],
        )?;
        Ok(())
    }

    // ─────────────────────────────────────────
    //  Stats
    // ─────────────────────────────────────────

    /// Obtiene estadísticas del sistema
    pub fn get_stats(&self) -> Result<SystemStats> {
        let total_files = self.count_files()?;
        let total_embedded = self.count_files_by_status("indexed")?;

        let total_categories: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM categories", [], |row| row.get(0))?;

        let total_watched: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM watched_folders", [], |row| row.get(0))?;

        let total_size: i64 = self
            .conn
            .query_row(
                "SELECT COALESCE(SUM(size), 0) FROM indexed_files",
                [],
                |row| row.get(0),
            )?;

        Ok(SystemStats {
            total_files,
            total_embedded,
            total_categories: total_categories as u64,
            total_watched_folders: total_watched as u64,
            database_size_bytes: 0, // Se calculará aparte
            indexed_files_size_bytes: total_size as u64,
        })
    }

    // ─────────────────────────────────────────
    //  Helper privado
    // ─────────────────────────────────────────

    fn row_to_indexed_file(&self, row: &rusqlite::Row) -> rusqlite::Result<IndexedFile> {
        let metadata_json: String = row.get(9)?;
        let created_str: String = row.get(10)?;
        let modified_str: String = row.get(11)?;
        let indexed_str: String = row.get(12)?;
        let status_str: String = row.get(13)?;
        let status_detail: Option<String> = row.get(14)?;

        Ok(IndexedFile {
            id: row.get(0)?,
            path: std::path::PathBuf::from(row.get::<_, String>(1)?),
            filename: row.get(2)?,
            extension: row.get(3)?,
            mime_type: row.get(4)?,
            size: row.get::<_, i64>(5)? as u64,
            content_hash: row.get(6)?,
            content_preview: row.get(7)?,
            content_full: row.get(8)?,
            metadata: serde_json::from_str(&metadata_json).unwrap_or_default(),
            created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
                .map(|d| d.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            modified_at: chrono::DateTime::parse_from_rfc3339(&modified_str)
                .map(|d| d.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            indexed_at: chrono::DateTime::parse_from_rfc3339(&indexed_str)
                .map(|d| d.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            index_status: IndexStatus::from_str_with_detail(&status_str, status_detail.as_deref()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_get_file() {
        let storage = SqliteStorage::in_memory().unwrap();

        let mut file = IndexedFile::new(std::path::PathBuf::from("/tmp/test.txt"));
        file.filename = "test.txt".to_string();
        file.extension = "txt".to_string();
        file.content_preview = "Hello world".to_string();
        file.index_status = IndexStatus::Indexed;

        storage.upsert_file(&file).unwrap();

        let retrieved = storage.get_file_by_id(&file.id).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().filename, "test.txt");
    }

    #[test]
    fn test_categories_and_virtual_files() {
        let storage = SqliteStorage::in_memory().unwrap();

        // Crear archivo
        let file = IndexedFile::new(std::path::PathBuf::from("/tmp/recibo.pdf"));
        storage.upsert_file(&file).unwrap();

        // Crear categoría
        let cat = Category::new("Recibos", "Recibos y facturas");
        storage.create_category(&cat).unwrap();

        // Asignar virtual
        let vf = VirtualFile {
            file_id: file.id.clone(),
            category_id: cat.id.clone(),
            virtual_name: "Recibo de Luz - Enero 2024".to_string(),
            notes: None,
            tags: vec!["luz".to_string(), "cfe".to_string()],
            sort_order: 0,
            auto_classified: true,
        };
        storage.set_virtual_file(&vf).unwrap();

        let results = storage.get_virtual_files_by_category(&cat.id).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.virtual_name, "Recibo de Luz - Enero 2024");
    }

    #[test]
    fn test_watched_folders() {
        let storage = SqliteStorage::in_memory().unwrap();

        let folder = WatchedFolder::new(
            std::path::PathBuf::from("/home/user/Documents"),
            "Documentos",
        );
        storage.add_watched_folder(&folder).unwrap();

        let folders = storage.get_watched_folders().unwrap();
        assert_eq!(folders.len(), 1);
        assert_eq!(folders[0].name, "Documentos");
    }
}
