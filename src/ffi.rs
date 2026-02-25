//! Capa FFI para comunicación con Flutter via dart:ffi
//!
//! Expone funciones C-compatible que Flutter puede llamar directamente.
//! Usa una instancia global del runtime de Tokio para operaciones async.
//!
//! Convención:
//! - Todas las funciones retornan un `*mut c_char` con JSON
//! - Las cadenas de entrada son `*const c_char`
//! - `ffi_free_string` debe llamarse para liberar las cadenas retornadas

use crate::config::SoasConfig;
use crate::embeddings::OllamaClient;
use crate::error::Result;
use crate::models::*;
use crate::storage::SqliteStorage;
use crate::vector_store::{InMemoryVectorStore, VectorStore};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::Mutex;
use tracing::error;

// ─────────────────────────────────────────────
//  Estado global
// ─────────────────────────────────────────────

struct SoasState {
    config: SoasConfig,
    storage: SqliteStorage,
    vector_store: InMemoryVectorStore,
    ollama: OllamaClient,
    runtime: tokio::runtime::Runtime,
}

static SOAS: Mutex<Option<SoasState>> = Mutex::new(None);

// ─────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────

/// Convierte un resultado a JSON string y lo retorna como C string
fn result_to_c_string<T: serde::Serialize>(result: Result<T>) -> *mut c_char {
    let json = match result {
        Ok(data) => {
            serde_json::json!({
                "success": true,
                "data": data
            })
        }
        Err(e) => {
            serde_json::json!({
                "success": false,
                "error": e.to_string()
            })
        }
    };

    let json_str = serde_json::to_string(&json).unwrap_or_else(|_| {
        r#"{"success": false, "error": "Error de serialización"}"#.to_string()
    });

    CString::new(json_str)
        .unwrap_or_else(|_| CString::new("null").unwrap())
        .into_raw()
}

/// Convierte un C string a &str
unsafe fn c_str_to_str<'a>(ptr: *const c_char) -> &'a str { unsafe {
    if ptr.is_null() {
        return "";
    }
    CStr::from_ptr(ptr).to_str().unwrap_or("")
}}

// ─────────────────────────────────────────────
//  Inicialización
// ─────────────────────────────────────────────

/// Inicializa SOAS con configuración por defecto o desde un path
///
/// # Safety
/// `config_path` puede ser null (usa config por defecto) o un path válido a un JSON de config
#[unsafe(no_mangle)]
pub unsafe extern "C" fn soas_init(config_path: *const c_char) -> *mut c_char { unsafe {
    let result = (|| -> Result<String> {
        // Inicializar logging
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::from_default_env()
                    .add_directive(tracing::Level::INFO.into()),
            )
            .try_init();

        let config = if config_path.is_null() {
            SoasConfig::default()
        } else {
            let path = c_str_to_str(config_path);
            SoasConfig::load(std::path::Path::new(path))?
        };

        // Crear directorio de datos
        std::fs::create_dir_all(&config.storage.data_dir)?;

        // Abrir SQLite
        let storage = SqliteStorage::open(&config.storage.db_path())?;

        // Abrir vector store
        let vector_store = InMemoryVectorStore::open(
            config.storage.vector_store_path(),
            config.ollama.embedding_dimensions,
        )?;

        // Crear cliente Ollama
        let ollama = OllamaClient::new(config.ollama.clone());

        // Crear runtime de Tokio
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| crate::error::SoasError::Other(e.to_string()))?;

        let mut state = SOAS.lock().unwrap();
        *state = Some(SoasState {
            config,
            storage,
            vector_store,
            ollama,
            runtime,
        });

        Ok("SOAS inicializado correctamente".to_string())
    })();

    result_to_c_string(result)
}}

/// Libera los recursos de SOAS
#[unsafe(no_mangle)]
pub extern "C" fn soas_destroy() {
    let mut state = SOAS.lock().unwrap();
    if let Some(ref s) = *state {
        // Guardar vectores pendientes
        let _ = s.vector_store.save();
    }
    *state = None;
}

// ─────────────────────────────────────────────
//  Carpetas monitoreadas
// ─────────────────────────────────────────────

/// Agrega una carpeta para indexar
///
/// # Safety
/// `path` y `name` deben ser strings C válidos
#[unsafe(no_mangle)]
pub unsafe extern "C" fn soas_add_folder(
    path: *const c_char,
    name: *const c_char,
) -> *mut c_char { unsafe {
    let path_str = c_str_to_str(path);
    let name_str = c_str_to_str(name);

    let result = (|| -> Result<WatchedFolder> {
        let state = SOAS.lock().unwrap();
        let state = state
            .as_ref()
            .ok_or_else(|| crate::error::SoasError::Other("SOAS no inicializado".into()))?;

        let folder =
            WatchedFolder::new(std::path::PathBuf::from(path_str), name_str);
        state.storage.add_watched_folder(&folder)?;

        Ok(folder)
    })();

    result_to_c_string(result)
}}

/// Lista las carpetas monitoreadas
#[unsafe(no_mangle)]
pub extern "C" fn soas_list_folders() -> *mut c_char {
    let result = (|| -> Result<Vec<WatchedFolder>> {
        let state = SOAS.lock().unwrap();
        let state = state
            .as_ref()
            .ok_or_else(|| crate::error::SoasError::Other("SOAS no inicializado".into()))?;

        state.storage.get_watched_folders()
    })();

    result_to_c_string(result)
}

/// Elimina una carpeta del monitoreo
///
/// # Safety
/// `folder_id` debe ser un string C válido
#[unsafe(no_mangle)]
pub unsafe extern "C" fn soas_remove_folder(folder_id: *const c_char) -> *mut c_char { unsafe {
    let id = c_str_to_str(folder_id);

    let result = (|| -> Result<bool> {
        let state = SOAS.lock().unwrap();
        let state = state
            .as_ref()
            .ok_or_else(|| crate::error::SoasError::Other("SOAS no inicializado".into()))?;

        state.storage.remove_watched_folder(id)?;
        Ok(true)
    })();

    result_to_c_string(result)
}}

// ─────────────────────────────────────────────
//  Indexación
// ─────────────────────────────────────────────

/// Escanea e indexa todas las carpetas monitoreadas
#[unsafe(no_mangle)]
pub extern "C" fn soas_scan_all() -> *mut c_char {
    let result = (|| -> Result<serde_json::Value> {
        let mut state = SOAS.lock().unwrap();
        let state = state
            .as_mut()
            .ok_or_else(|| crate::error::SoasError::Other("SOAS no inicializado".into()))?;

        let folders = state.storage.get_watched_folders()?;

        let config = state.config.indexer.clone();
        let mut total_new = 0u64;
        let mut total_updated = 0u64;
        let mut total_deleted = 0u64;
        let mut total_failed = 0u64;

        for folder in &folders {
            if !folder.active {
                continue;
            }

            let mut pipeline = crate::indexer::IndexPipeline::new(
                &state.storage,
                &mut state.vector_store,
                &state.ollama,
                config.clone(),
            );

            match state.runtime.block_on(pipeline.scan_folder(folder, None)) {
                Ok(result) => {
                    total_new += result.new_files;
                    total_updated += result.updated_files;
                    total_deleted += result.deleted_files;
                    total_failed += result.failed_files;
                }
                Err(e) => {
                    error!("Error escaneando {:?}: {}", folder.path, e);
                    total_failed += 1;
                }
            }
        }

        Ok(serde_json::json!({
            "new_files": total_new,
            "updated_files": total_updated,
            "deleted_files": total_deleted,
            "failed_files": total_failed,
        }))
    })();

    result_to_c_string(result)
}

// ─────────────────────────────────────────────
//  Búsqueda
// ─────────────────────────────────────────────

/// Realiza una búsqueda semántica
///
/// # Safety
/// `query_json` debe ser un string C válido con un JSON de SearchQuery
#[unsafe(no_mangle)]
pub unsafe extern "C" fn soas_search(query_json: *const c_char) -> *mut c_char { unsafe {
    let json_str = c_str_to_str(query_json);

    let result = (|| -> Result<Vec<SearchResult>> {
        let query: SearchQuery = serde_json::from_str(json_str)?;

        let state = SOAS.lock().unwrap();
        let state = state
            .as_ref()
            .ok_or_else(|| crate::error::SoasError::Other("SOAS no inicializado".into()))?;

        let engine = crate::search::SearchEngine::new(
            &state.storage,
            &state.vector_store,
            &state.ollama,
            state.config.search.clone(),
        );

        state.runtime.block_on(engine.search(&query))
    })();

    result_to_c_string(result)
}}

/// Búsqueda rápida con solo texto (wrapper simple)
///
/// # Safety
/// `query_text` debe ser un string C válido
#[unsafe(no_mangle)]
pub unsafe extern "C" fn soas_quick_search(query_text: *const c_char) -> *mut c_char { unsafe {
    let text = c_str_to_str(query_text);

    let result = (|| -> Result<Vec<SearchResult>> {
        let query = SearchQuery::new(text);

        let state = SOAS.lock().unwrap();
        let state = state
            .as_ref()
            .ok_or_else(|| crate::error::SoasError::Other("SOAS no inicializado".into()))?;

        let engine = crate::search::SearchEngine::new(
            &state.storage,
            &state.vector_store,
            &state.ollama,
            state.config.search.clone(),
        );

        state.runtime.block_on(engine.search(&query))
    })();

    result_to_c_string(result)
}}

// ─────────────────────────────────────────────
//  Categorías y organización virtual
// ─────────────────────────────────────────────

/// Obtiene el árbol de categorías
#[unsafe(no_mangle)]
pub extern "C" fn soas_get_categories() -> *mut c_char {
    let result = (|| -> Result<Vec<crate::virtual_fs::manager::CategoryTree>> {
        let state = SOAS.lock().unwrap();
        let state = state
            .as_ref()
            .ok_or_else(|| crate::error::SoasError::Other("SOAS no inicializado".into()))?;

        let manager = crate::virtual_fs::VirtualFsManager::new(&state.storage, &state.ollama);
        manager.get_category_tree()
    })();

    result_to_c_string(result)
}

/// Crea una categoría
///
/// # Safety
/// `name` y `description` deben ser strings C válidos. `parent_id` puede ser null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn soas_create_category(
    name: *const c_char,
    description: *const c_char,
    parent_id: *const c_char,
) -> *mut c_char { unsafe {
    let name = c_str_to_str(name);
    let desc = c_str_to_str(description);
    let parent = if parent_id.is_null() {
        None
    } else {
        Some(c_str_to_str(parent_id))
    };

    let result = (|| -> Result<Category> {
        let state = SOAS.lock().unwrap();
        let state = state
            .as_ref()
            .ok_or_else(|| crate::error::SoasError::Other("SOAS no inicializado".into()))?;

        let manager = crate::virtual_fs::VirtualFsManager::new(&state.storage, &state.ollama);
        manager.create_category(name, desc, parent)
    })();

    result_to_c_string(result)
}}

/// Asigna un archivo a una categoría con nombre virtual
///
/// # Safety
/// Todos los parámetros deben ser strings C válidos
#[unsafe(no_mangle)]
pub unsafe extern "C" fn soas_assign_file(
    file_id: *const c_char,
    category_id: *const c_char,
    virtual_name: *const c_char,
) -> *mut c_char { unsafe {
    let file_id = c_str_to_str(file_id);
    let cat_id = c_str_to_str(category_id);
    let vname = c_str_to_str(virtual_name);

    let result = (|| -> Result<VirtualFile> {
        let state = SOAS.lock().unwrap();
        let state = state
            .as_ref()
            .ok_or_else(|| crate::error::SoasError::Other("SOAS no inicializado".into()))?;

        let manager = crate::virtual_fs::VirtualFsManager::new(&state.storage, &state.ollama);
        manager.assign_file(file_id, cat_id, vname)
    })();

    result_to_c_string(result)
}}

/// Obtiene los archivos de una categoría
///
/// # Safety
/// `category_id` debe ser un string C válido
#[unsafe(no_mangle)]
pub unsafe extern "C" fn soas_get_category_files(category_id: *const c_char) -> *mut c_char { unsafe {
    let cat_id = c_str_to_str(category_id);

    let result = (|| -> Result<Vec<serde_json::Value>> {
        let state = SOAS.lock().unwrap();
        let state = state
            .as_ref()
            .ok_or_else(|| crate::error::SoasError::Other("SOAS no inicializado".into()))?;

        let manager = crate::virtual_fs::VirtualFsManager::new(&state.storage, &state.ollama);
        let files = manager.list_category_files(cat_id)?;

        Ok(files
            .into_iter()
            .map(|(vf, file)| {
                serde_json::json!({
                    "virtual": vf,
                    "file": file,
                })
            })
            .collect())
    })();

    result_to_c_string(result)
}}

// ─────────────────────────────────────────────
//  Estadísticas
// ─────────────────────────────────────────────

/// Obtiene estadísticas del sistema
#[unsafe(no_mangle)]
pub extern "C" fn soas_get_stats() -> *mut c_char {
    let result = (|| -> Result<SystemStats> {
        let state = SOAS.lock().unwrap();
        let state = state
            .as_ref()
            .ok_or_else(|| crate::error::SoasError::Other("SOAS no inicializado".into()))?;

        state.storage.get_stats()
    })();

    result_to_c_string(result)
}

/// Verifica la salud del sistema (conectividad a Ollama, etc.)
#[unsafe(no_mangle)]
pub extern "C" fn soas_health_check() -> *mut c_char {
    let result = (|| -> Result<serde_json::Value> {
        let state = SOAS.lock().unwrap();
        let state = state
            .as_ref()
            .ok_or_else(|| crate::error::SoasError::Other("SOAS no inicializado".into()))?;

        let ollama_ok = state.runtime.block_on(state.ollama.health_check())?;

        Ok(serde_json::json!({
            "database": true,
            "vector_store": true,
            "ollama": ollama_ok,
            "vector_count": state.vector_store.len(),
        }))
    })();

    result_to_c_string(result)
}

// ─────────────────────────────────────────────
//  Exploración del sistema de archivos real
// ─────────────────────────────────────────────

/// Entrada de directorio (archivo o carpeta del SO)
#[derive(serde::Serialize)]
struct DirEntry {
    name: String,
    path: String,
    is_dir: bool,
    is_symlink: bool,
    size: u64,
    modified: Option<String>,
    extension: Option<String>,
    is_indexed: bool,
}

/// Lista el contenido de un directorio real del SO
///
/// # Safety
/// `path` debe ser un string C válido con una ruta absoluta
#[unsafe(no_mangle)]
pub unsafe extern "C" fn soas_browse_directory(path: *const c_char) -> *mut c_char { unsafe {
    let path_str = c_str_to_str(path);

    let result = (|| -> crate::error::Result<serde_json::Value> {
        if path_str.is_empty() {
            return Err(crate::error::SoasError::Other("Ruta vacía".into()));
        }

        let dir_path = std::path::Path::new(path_str);
        if !dir_path.exists() {
            return Err(crate::error::SoasError::Other(format!("Ruta no existe: {}", path_str)));
        }

        // Obtener hashes de archivos indexados para marcar cuáles están indexados
        let indexed_paths: std::collections::HashSet<String> = {
            let state = SOAS.lock().unwrap();
            if let Some(ref s) = *state {
                s.storage
                    .get_all_file_paths_and_hashes()
                    .unwrap_or_default()
                    .into_iter()
                    .map(|(_, p, _)| p)
                    .collect()
            } else {
                std::collections::HashSet::new()
            }
        };

        let mut entries: Vec<DirEntry> = Vec::new();
        let read_dir = std::fs::read_dir(dir_path)
            .map_err(|e| crate::error::SoasError::Other(e.to_string()))?;

        for entry in read_dir.flatten() {
            let meta = entry.metadata().ok();
            let file_type = entry.file_type().ok();
            let is_dir = meta.as_ref().map(|m| m.is_dir()).unwrap_or(false);
            let is_symlink = file_type.as_ref().map(|t| t.is_symlink()).unwrap_or(false);
            let size = if is_dir { 0 } else { meta.as_ref().map(|m| m.len()).unwrap_or(0) };
            let modified = meta
                .as_ref()
                .and_then(|m| m.modified().ok())
                .map(|t| {
                    let dt: chrono::DateTime<chrono::Utc> = t.into();
                    dt.to_rfc3339()
                });
            let path_owned = entry.path();
            let path_string = path_owned.to_string_lossy().to_string();
            let extension = if is_dir {
                None
            } else {
                path_owned.extension().map(|e| e.to_string_lossy().to_string())
            };
            let is_indexed = indexed_paths.contains(&path_string);

            entries.push(DirEntry {
                name: entry.file_name().to_string_lossy().to_string(),
                path: path_string,
                is_dir,
                is_symlink,
                size,
                modified,
                extension,
                is_indexed,
            });
        }

        // Carpetas primero, luego archivos, ambos en orden alfabético
        entries.sort_by(|a, b| match (a.is_dir, b.is_dir) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
        });

        let total = entries.len();
        let canonical = dir_path
            .canonicalize()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| path_str.to_string());

        let parent = dir_path
            .parent()
            .map(|p| p.to_string_lossy().to_string());

        Ok(serde_json::json!({
            "path": canonical,
            "parent": parent,
            "entries": entries,
            "total": total,
        }))
    })();

    result_to_c_string(result)
}}

/// Retorna el directorio home del usuario
#[unsafe(no_mangle)]
pub extern "C" fn soas_get_home_dir() -> *mut c_char {
    let result: crate::error::Result<String> = {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| "/".to_string());
        Ok(home)
    };
    result_to_c_string(result)
}

/// Retorna los directorios especiales del usuario (Documentos, Descargas, etc.)
#[unsafe(no_mangle)]
pub extern "C" fn soas_get_special_dirs() -> *mut c_char {
    let result: crate::error::Result<serde_json::Value> = {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| "/".to_string());

        let candidates = [
            ("home", home.clone()),
            ("documents", format!("{}/Documents", home)),
            ("downloads", format!("{}/Downloads", home)),
            ("desktop", format!("{}/Desktop", home)),
            ("pictures", format!("{}/Pictures", home)),
            ("videos", format!("{}/Videos", home)),
            ("music", format!("{}/Music", home)),
        ];

        let dirs: serde_json::Value = candidates
            .iter()
            .filter_map(|(key, path)| {
                if std::path::Path::new(path).is_dir() {
                    Some(serde_json::json!({ "key": key, "path": path }))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .into();

        Ok(dirs)
    };
    result_to_c_string(result)
}

/// Retorna información sobre una ruta (archivo o carpeta)
///
/// # Safety
/// `path` debe ser un string C válido
#[unsafe(no_mangle)]
pub unsafe extern "C" fn soas_path_info(path: *const c_char) -> *mut c_char { unsafe {
    let path_str = c_str_to_str(path);

    let result = (|| -> crate::error::Result<serde_json::Value> {
        let p = std::path::Path::new(path_str);
        let exists = p.exists();

        if !exists {
            return Ok(serde_json::json!({ "exists": false, "path": path_str }));
        }

        let meta = std::fs::metadata(p)
            .map_err(|e| crate::error::SoasError::Other(e.to_string()))?;

        let is_dir = meta.is_dir();
        let size = if is_dir { 0u64 } else { meta.len() };
        let modified: Option<String> = meta
            .modified()
            .ok()
            .map(|t| {
                let dt: chrono::DateTime<chrono::Utc> = t.into();
                dt.to_rfc3339()
            });
        let created: Option<String> = meta
            .created()
            .ok()
            .map(|t| {
                let dt: chrono::DateTime<chrono::Utc> = t.into();
                dt.to_rfc3339()
            });

        let canonical = p
            .canonicalize()
            .map(|c| c.to_string_lossy().to_string())
            .unwrap_or_else(|_| path_str.to_string());

        Ok(serde_json::json!({
            "exists": true,
            "path": canonical,
            "name": p.file_name().map(|n| n.to_string_lossy()),
            "extension": p.extension().map(|e| e.to_string_lossy()),
            "parent": p.parent().map(|pp| pp.to_string_lossy()),
            "is_dir": is_dir,
            "is_file": meta.is_file(),
            "is_symlink": meta.is_symlink(),
            "size": size,
            "modified": modified,
            "created": created,
        }))
    })();

    result_to_c_string(result)
}}

/// Verifica si una ruta existe
///
/// # Safety
/// `path` debe ser un string C válido
#[unsafe(no_mangle)]
pub unsafe extern "C" fn soas_path_exists(path: *const c_char) -> *mut c_char { unsafe {
    let path_str = c_str_to_str(path);
    let exists = std::path::Path::new(path_str).exists();
    result_to_c_string(Ok::<_, crate::error::SoasError>(exists))
}}

// ─────────────────────────────────────────────
//  Spotlight search (rápido, por nombre de archivo)
// ─────────────────────────────────────────────

/// Resultado de búsqueda tipo Spotlight
#[derive(serde::Serialize)]
struct SpotlightResult {
    id: String,
    filename: String,
    path: String,
    extension: String,
    mime_type: String,
    size: u64,
    modified: Option<String>,
    /// Nivel de coincidencia: 1=exacto 2=empieza 3=palabra 4=contiene 5=ruta
    match_rank: u8,
    is_dir: bool,
}

/// Búsqueda rápida tipo Spotlight: busca en nombres de archivos indexados.
///
/// Devuelve resultados en < 5 ms para colecciones de hasta 500 K archivos.
/// Orden de relevancia: exacto → empieza_por → empieza_palabra → contiene → ruta.
///
/// # Safety
/// `query` debe ser un string C válido (puede ser parcial, p.ej. "reporte")
#[unsafe(no_mangle)]
pub unsafe extern "C" fn soas_spotlight_search(
    query: *const c_char,
    limit: i64,
) -> *mut c_char { unsafe {
    let query_str = c_str_to_str(query);
    let lim = if limit <= 0 { 30 } else { limit as usize };

    let result = (|| -> crate::error::Result<serde_json::Value> {
        if query_str.trim().is_empty() {
            return Ok(serde_json::json!({ "results": [], "query": query_str, "total": 0 }));
        }

        let state = SOAS.lock().unwrap();
        let state = state
            .as_ref()
            .ok_or_else(|| crate::error::SoasError::Other("SOAS no inicializado".into()))?;

        let start = std::time::Instant::now();
        let matches = state.storage.spotlight_search(query_str, lim)?;
        let elapsed_ms = start.elapsed().as_millis();

        let results: Vec<SpotlightResult> = matches
            .into_iter()
            .map(|(file, rank)| {
                let modified = {
                    let dt: chrono::DateTime<chrono::Utc> = file.modified_at;
                    Some(dt.to_rfc3339())
                };
                SpotlightResult {
                    id: file.id,
                    filename: file.filename,
                    path: file.path.to_string_lossy().to_string(),
                    extension: file.extension,
                    mime_type: file.mime_type,
                    size: file.size,
                    modified,
                    match_rank: rank,
                    is_dir: false,
                }
            })
            .collect();

        let total = results.len();
        Ok(serde_json::json!({
            "results": results,
            "query": query_str,
            "total": total,
            "elapsed_ms": elapsed_ms,
        }))
    })();

    result_to_c_string(result)
}}

// ─────────────────────────────────────────────
//  Consulta de archivos indexados
// ─────────────────────────────────────────────

/// Lista archivos indexados con paginación y filtros
///
/// `filter_json` puede contener:
/// - `limit` (u64, por defecto 50)
/// - `offset` (u64, por defecto 0)
/// - `extension` (string opcional)
/// - `path_prefix` (string opcional, filtra por prefijo de ruta)
/// - `sort_by` ("modified_at" | "name" | "size" | "created_at" | "indexed_at")
///
/// # Safety
/// `filter_json` debe ser un string C válido
#[unsafe(no_mangle)]
pub unsafe extern "C" fn soas_list_files(filter_json: *const c_char) -> *mut c_char { unsafe {
    let json_str = c_str_to_str(filter_json);

    let result = (|| -> crate::error::Result<serde_json::Value> {
        let filter: serde_json::Value = if json_str.is_empty() {
            serde_json::json!({})
        } else {
            serde_json::from_str(json_str)?
        };

        let limit = filter["limit"].as_u64().unwrap_or(50) as usize;
        let offset = filter["offset"].as_u64().unwrap_or(0) as usize;
        let extension = filter["extension"].as_str();
        let path_prefix = filter["path_prefix"].as_str();
        let sort_by = filter["sort_by"].as_str();

        let state = SOAS.lock().unwrap();
        let state = state
            .as_ref()
            .ok_or_else(|| crate::error::SoasError::Other("SOAS no inicializado".into()))?;

        let files = state.storage.get_files_paginated(
            limit, offset, extension, path_prefix, sort_by,
        )?;

        let total = state.storage.count_files_filtered(extension, path_prefix)?;

        Ok(serde_json::json!({
            "files": files,
            "total": total,
            "limit": limit,
            "offset": offset,
        }))
    })();

    result_to_c_string(result)
}}

/// Obtiene un archivo indexado por su ID
///
/// # Safety
/// `file_id` debe ser un string C válido
#[unsafe(no_mangle)]
pub unsafe extern "C" fn soas_get_file(file_id: *const c_char) -> *mut c_char { unsafe {
    let id = c_str_to_str(file_id);

    let result = (|| -> crate::error::Result<Option<crate::models::IndexedFile>> {
        let state = SOAS.lock().unwrap();
        let state = state
            .as_ref()
            .ok_or_else(|| crate::error::SoasError::Other("SOAS no inicializado".into()))?;

        state.storage.get_file_by_id(id)
    })();

    result_to_c_string(result)
}}

/// Lista archivos indexados bajo una ruta (búsqueda por prefijo de path)
///
/// # Safety
/// `path` debe ser un string C válido
#[unsafe(no_mangle)]
pub unsafe extern "C" fn soas_get_files_in_path(
    path: *const c_char,
    limit: i64,
    offset: i64,
) -> *mut c_char { unsafe {
    let path_str = c_str_to_str(path);
    let lim = if limit <= 0 { 200 } else { limit as usize };
    let off = if offset < 0 { 0 } else { offset as usize };

    let result = (|| -> crate::error::Result<serde_json::Value> {
        let state = SOAS.lock().unwrap();
        let state = state
            .as_ref()
            .ok_or_else(|| crate::error::SoasError::Other("SOAS no inicializado".into()))?;

        let files = state.storage.get_files_paginated(
            lim, off, None, Some(path_str), Some("name"),
        )?;
        let total = state.storage.count_files_filtered(None, Some(path_str))?;

        Ok(serde_json::json!({
            "files": files,
            "total": total,
            "path": path_str,
        }))
    })();

    result_to_c_string(result)
}}

/// Archivos modificados más recientemente
///
/// # Safety
/// Siempre seguro de llamar cuando SOAS está inicializado
#[unsafe(no_mangle)]
pub extern "C" fn soas_get_recent_files(limit: i64) -> *mut c_char {
    let lim = if limit <= 0 { 20 } else { limit as usize };

    let result = (|| -> crate::error::Result<Vec<crate::models::IndexedFile>> {
        let state = SOAS.lock().unwrap();
        let state = state
            .as_ref()
            .ok_or_else(|| crate::error::SoasError::Other("SOAS no inicializado".into()))?;

        state.storage.get_recent_files(lim)
    })();

    result_to_c_string(result)
}

/// Archivos filtrados por extensiones
///
/// `extensions_json` es un array JSON de extensiones, e.g. `["pdf","docx"]`
///
/// # Safety
/// `extensions_json` debe ser un string C válido
#[unsafe(no_mangle)]
pub unsafe extern "C" fn soas_get_files_by_extension(
    extensions_json: *const c_char,
    limit: i64,
    offset: i64,
) -> *mut c_char { unsafe {
    let json_str = c_str_to_str(extensions_json);
    let lim = if limit <= 0 { 100 } else { limit as usize };
    let off = if offset < 0 { 0 } else { offset as usize };

    let result = (|| -> crate::error::Result<serde_json::Value> {
        let exts: Vec<String> = serde_json::from_str(json_str).unwrap_or_default();
        let ext_refs: Vec<&str> = exts.iter().map(|s| s.as_str()).collect();

        let state = SOAS.lock().unwrap();
        let state = state
            .as_ref()
            .ok_or_else(|| crate::error::SoasError::Other("SOAS no inicializado".into()))?;

        let files = state.storage.get_files_by_extensions(&ext_refs, lim, off)?;

        Ok(serde_json::json!({
            "files": files,
            "extensions": exts,
        }))
    })();

    result_to_c_string(result)
}}

/// Lista todas las extensiones únicas presentes en el índice
#[unsafe(no_mangle)]
pub extern "C" fn soas_get_distinct_extensions() -> *mut c_char {
    let result = (|| -> crate::error::Result<Vec<String>> {
        let state = SOAS.lock().unwrap();
        let state = state
            .as_ref()
            .ok_or_else(|| crate::error::SoasError::Other("SOAS no inicializado".into()))?;

        state.storage.get_distinct_extensions()
    })();

    result_to_c_string(result)
}

// ─────────────────────────────────────────────
//  Operaciones sobre archivos
// ─────────────────────────────────────────────

/// Abre un archivo con la aplicación predeterminada del sistema
///
/// # Safety
/// `path` debe ser un string C válido
#[unsafe(no_mangle)]
pub unsafe extern "C" fn soas_open_file(path: *const c_char) -> *mut c_char { unsafe {
    let path_str = c_str_to_str(path);

    let result = (|| -> crate::error::Result<bool> {
        if !std::path::Path::new(path_str).exists() {
            return Err(crate::error::SoasError::Other(
                format!("Archivo no encontrado: {}", path_str)
            ));
        }

        // Linux: xdg-open | macOS: open | Windows: start
        let opener = if cfg!(target_os = "macos") {
            "open"
        } else if cfg!(target_os = "windows") {
            "explorer"
        } else {
            "xdg-open"
        };

        std::process::Command::new(opener)
            .arg(path_str)
            .spawn()
            .map_err(|e| crate::error::SoasError::Other(e.to_string()))?;

        Ok(true)
    })();

    result_to_c_string(result)
}}

/// Muestra un archivo en el gestor de archivos del sistema (resalta el archivo)
///
/// # Safety
/// `path` debe ser un string C válido
#[unsafe(no_mangle)]
pub unsafe extern "C" fn soas_reveal_in_folder(path: *const c_char) -> *mut c_char { unsafe {
    let path_str = c_str_to_str(path);

    let result = (|| -> crate::error::Result<bool> {
        let p = std::path::Path::new(path_str);
        let target = if p.is_file() {
            p.parent().unwrap_or(p).to_string_lossy().to_string()
        } else {
            path_str.to_string()
        };

        let opener = if cfg!(target_os = "macos") {
            "open"
        } else if cfg!(target_os = "windows") {
            "explorer"
        } else {
            "xdg-open"
        };

        std::process::Command::new(opener)
            .arg(&target)
            .spawn()
            .map_err(|e| crate::error::SoasError::Other(e.to_string()))?;

        Ok(true)
    })();

    result_to_c_string(result)
}}

/// Elimina un archivo del índice (NO elimina el archivo físico)
///
/// # Safety
/// `file_id` debe ser un string C válido
#[unsafe(no_mangle)]
pub unsafe extern "C" fn soas_delete_file_from_index(file_id: *const c_char) -> *mut c_char { unsafe {
    let id = c_str_to_str(file_id);

    let result = (|| -> crate::error::Result<bool> {
        let mut state = SOAS.lock().unwrap();
        let state = state
            .as_mut()
            .ok_or_else(|| crate::error::SoasError::Other("SOAS no inicializado".into()))?;

        state.storage.delete_file(id)?;
        let _ = state.vector_store.remove(id);
        Ok(true)
    })();

    result_to_c_string(result)
}}

/// Actualiza los ajustes de una carpeta monitoreada
///
/// `update_json` puede contener: `name`, `recursive` (bool), `active` (bool),
/// `exclude_patterns` (array de strings)
///
/// # Safety
/// `folder_id` y `update_json` deben ser strings C válidos
#[unsafe(no_mangle)]
pub unsafe extern "C" fn soas_update_folder(
    folder_id: *const c_char,
    update_json: *const c_char,
) -> *mut c_char { unsafe {
    let id = c_str_to_str(folder_id);
    let json_str = c_str_to_str(update_json);

    let result = (|| -> crate::error::Result<crate::models::WatchedFolder> {
        let patch: serde_json::Value = serde_json::from_str(json_str)?;

        let state = SOAS.lock().unwrap();
        let state = state
            .as_ref()
            .ok_or_else(|| crate::error::SoasError::Other("SOAS no inicializado".into()))?;

        let mut folders = state.storage.get_watched_folders()?;
        let folder = folders
            .iter_mut()
            .find(|f| f.id == id)
            .ok_or_else(|| crate::error::SoasError::Other(format!("Carpeta no encontrada: {}", id)))?;

        if let Some(name) = patch["name"].as_str() {
            folder.name = name.to_string();
        }
        if let Some(recursive) = patch["recursive"].as_bool() {
            folder.recursive = recursive;
        }
        if let Some(active) = patch["active"].as_bool() {
            folder.active = active;
        }
        if let Some(patterns) = patch["exclude_patterns"].as_array() {
            folder.exclude_patterns = patterns
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
        }

        state.storage.update_watched_folder(folder)?;
        Ok(folder.clone())
    })();

    result_to_c_string(result)
}}

// ─────────────────────────────────────────────
//  Gestión del índice
// ─────────────────────────────────────────────

/// Re-indexa un archivo individual por su ruta
///
/// # Safety
/// `path` debe ser un string C válido con una ruta absoluta
#[unsafe(no_mangle)]
pub unsafe extern "C" fn soas_reindex_file(path: *const c_char) -> *mut c_char { unsafe {
    let path_str = c_str_to_str(path);

    let result = (|| -> crate::error::Result<serde_json::Value> {
        let p = std::path::Path::new(path_str);
        if !p.exists() {
            return Err(crate::error::SoasError::Other(
                format!("Archivo no encontrado: {}", path_str)
            ));
        }
        if !p.is_file() {
            return Err(crate::error::SoasError::Other(
                format!("La ruta no es un archivo: {}", path_str)
            ));
        }

        let mut state = SOAS.lock().unwrap();
        let state = state
            .as_mut()
            .ok_or_else(|| crate::error::SoasError::Other("SOAS no inicializado".into()))?;

        // Si ya existe en el índice, obtener su ID para actualizar en lugar de crear nuevo
        let existing_id: Option<String> = state
            .storage
            .get_file_by_path(p)
            .ok()
            .flatten()
            .map(|f| {
                let _ = state.vector_store.remove(&f.id);
                f.id
            });

        let config = state.config.indexer.clone();
        let mut pipeline = crate::indexer::IndexPipeline::new(
            &state.storage,
            &mut state.vector_store,
            &state.ollama,
            config,
        );

        let indexed_file = state.runtime.block_on(
            pipeline.process_file(p, existing_id.as_deref())
        )?;

        Ok(serde_json::json!({
            "path": path_str,
            "file_id": indexed_file.id,
            "status": indexed_file.index_status.as_str(),
            "filename": indexed_file.filename,
        }))
    })();

    result_to_c_string(result)
}}

/// Escanea una carpeta específica por su ID
///
/// # Safety
/// `folder_id` debe ser un string C válido
#[unsafe(no_mangle)]
pub unsafe extern "C" fn soas_scan_folder(folder_id: *const c_char) -> *mut c_char { unsafe {
    let id = c_str_to_str(folder_id);

    let result = (|| -> crate::error::Result<serde_json::Value> {
        let mut state = SOAS.lock().unwrap();
        let state = state
            .as_mut()
            .ok_or_else(|| crate::error::SoasError::Other("SOAS no inicializado".into()))?;

        let folders = state.storage.get_watched_folders()?;
        let folder = folders
            .iter()
            .find(|f| f.id == id)
            .ok_or_else(|| crate::error::SoasError::Other(format!("Carpeta no encontrada: {}", id)))?
            .clone();

        let config = state.config.indexer.clone();
        let mut pipeline = crate::indexer::IndexPipeline::new(
            &state.storage,
            &mut state.vector_store,
            &state.ollama,
            config,
        );

        let scan_result = state.runtime.block_on(pipeline.scan_folder(&folder, None))?;

        Ok(serde_json::json!({
            "folder_id": id,
            "folder_path": folder.path.to_string_lossy(),
            "new_files": scan_result.new_files,
            "updated_files": scan_result.updated_files,
            "deleted_files": scan_result.deleted_files,
            "failed_files": scan_result.failed_files,
        }))
    })();

    result_to_c_string(result)
}}

// ─────────────────────────────────────────────
//  Gestión de memoria
// ─────────────────────────────────────────────

/// Libera una cadena retornada por SOAS
///
/// # Safety
/// `ptr` debe ser un puntero a una cadena retornada por una función de SOAS
#[unsafe(no_mangle)]
pub unsafe extern "C" fn soas_free_string(ptr: *mut c_char) { unsafe {
    if !ptr.is_null() {
        let _ = CString::from_raw(ptr);
    }
}}
