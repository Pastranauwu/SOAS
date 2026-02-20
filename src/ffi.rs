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
