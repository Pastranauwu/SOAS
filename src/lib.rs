//! # SOAS Core
//!
//! Sistema Inteligente de Búsqueda y Organización de Archivos.
//!
//! SOAS indexa archivos de carpetas seleccionadas, genera embeddings semánticos
//! usando Ollama (local), y permite buscar archivos con lenguaje natural.
//!
//! ## Arquitectura
//!
//! - **storage**: SQLite para metadatos, FTS5 para búsqueda full-text
//! - **vector_store**: Almacén de vectores embebido (similitud coseno)
//! - **embeddings**: Cliente Ollama para generar embeddings y consultas LLM
//! - **content**: Extractores de contenido (PDF, DOCX, TXT, CSV, imágenes)
//! - **indexer**: Pipeline de indexación y watcher de filesystem
//! - **search**: Motor de búsqueda híbrido (semántico + FTS)
//! - **virtual_fs**: Sistema de archivos virtual (categorías, nombres virtuales)
//! - **ffi**: Capa C-compatible para integración con Flutter
//!
//! ## Ejemplo de uso (desde Rust)
//!
//! ```no_run
//! use soas_core::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let mut soas = Soas::new(SoasConfig::default())?;
//!
//!     // Agregar carpeta
//!     soas.add_folder("/home/user/Documents", "Documentos").await?;
//!
//!     // Indexar
//!     soas.scan_all().await?;
//!
//!     // Buscar
//!     let results = soas.search("recibo de luz 2017").await?;
//!     for r in results {
//!         println!("{} (score: {:.2})", r.file.filename, r.score);
//!     }
//!
//!     Ok(())
//! }
//! ```

pub mod config;
pub mod content;
pub mod embeddings;
pub mod error;
pub mod ffi;
pub mod indexer;
pub mod models;
pub mod search;
pub mod storage;
pub mod vector_store;
pub mod virtual_fs;

use config::SoasConfig;
use embeddings::OllamaClient;
use error::{Result, SoasError};
use indexer::IndexPipeline;
use models::*;
use search::SearchEngine;
use storage::SqliteStorage;
use vector_store::{InMemoryVectorStore, VectorStore};
use virtual_fs::VirtualFsManager;

/// Prelude: importa los tipos más usados
pub mod prelude {
    pub use crate::config::SoasConfig;
    pub use crate::error::{Result, SoasError};
    pub use crate::models::*;
    pub use crate::Soas;
}

/// Punto de entrada principal del sistema SOAS
pub struct Soas {
    config: SoasConfig,
    storage: SqliteStorage,
    vector_store: InMemoryVectorStore,
    ollama: OllamaClient,
}

impl Soas {
    /// Crea una nueva instancia de SOAS con la configuración dada
    pub fn new(config: SoasConfig) -> Result<Self> {
        // Crear directorio de datos
        std::fs::create_dir_all(&config.storage.data_dir)?;

        let storage = SqliteStorage::open(&config.storage.db_path())?;
        let vector_store = InMemoryVectorStore::open(
            config.storage.vector_store_path(),
            config.ollama.embedding_dimensions,
        )?;
        let ollama = OllamaClient::new(config.ollama.clone());

        Ok(Self {
            config,
            storage,
            vector_store,
            ollama,
        })
    }

    /// Crea una instancia de SOAS en memoria (para tests)
    pub fn in_memory() -> Result<Self> {
        let config = SoasConfig::default();
        let storage = SqliteStorage::in_memory()?;
        let vector_store = InMemoryVectorStore::new(
            std::path::PathBuf::from("/tmp/soas_test_vectors.bin"),
            config.ollama.embedding_dimensions,
        );
        let ollama = OllamaClient::new(config.ollama.clone());

        Ok(Self {
            config,
            storage,
            vector_store,
            ollama,
        })
    }

    /// Verifica que Ollama esté disponible
    pub async fn check_ollama(&self) -> Result<bool> {
        self.ollama.health_check().await
    }

    /// Lista los modelos disponibles en Ollama
    pub async fn list_ollama_models(&self) -> Result<Vec<String>> {
        self.ollama.list_models().await
    }

    // ─── Carpetas ───────────────────────────

    /// Agrega una carpeta para indexar
    pub async fn add_folder(&self, path: &str, name: &str) -> Result<WatchedFolder> {
        let folder_path = std::path::PathBuf::from(path);
        if !folder_path.exists() {
            return Err(SoasError::FileNotFound(path.to_string()));
        }
        let folder = WatchedFolder::new(folder_path, name);
        self.storage.add_watched_folder(&folder)?;
        Ok(folder)
    }

    /// Lista las carpetas monitoreadas
    pub fn list_folders(&self) -> Result<Vec<WatchedFolder>> {
        self.storage.get_watched_folders()
    }

    /// Elimina una carpeta del monitoreo y todos sus archivos del índice
    pub fn remove_folder(&mut self, folder_id: &str) -> Result<u64> {
        // Obtener los datos de la carpeta antes de borrarla
        let folders = self.storage.get_watched_folders()?;
        let folder = folders.iter().find(|f| f.id == folder_id);

        let mut removed_files = 0u64;

        if let Some(folder) = folder {
            let prefix = folder.path.to_string_lossy().to_string();

            // Obtener IDs de archivos bajo esta carpeta para limpiar vectores
            let all_files = self.storage.get_all_file_paths_and_hashes()?;
            for (file_id, path, _) in &all_files {
                if path.starts_with(&prefix) {
                    let _ = self.vector_store.remove(file_id);
                    removed_files += 1;
                }
            }

            // Borrar archivos de la DB
            self.storage.delete_files_by_path_prefix(&prefix)?;
        }

        // Borrar la carpeta de watched
        self.storage.remove_watched_folder(folder_id)?;

        // Persistir vector store limpio
        if removed_files > 0 {
            self.vector_store.save()?;
        }

        tracing::info!("Carpeta eliminada: {} archivos removidos del índice", removed_files);
        Ok(removed_files)
    }

    /// Elimina un archivo específico del índice (DB + vector)
    pub fn remove_file(&mut self, file_id: &str) -> Result<bool> {
        // Verificar que existe
        let exists = self.storage.get_file_by_id(file_id)?.is_some();
        if !exists {
            return Ok(false);
        }

        // Borrar vector
        let _ = self.vector_store.remove(file_id);

        // Borrar de DB (cascade borra virtual_files también)
        self.storage.delete_file(file_id)?;

        self.vector_store.save()?;
        Ok(true)
    }

    /// Lista todos los archivos indexados
    pub fn list_files(&self) -> Result<Vec<IndexedFile>> {
        self.storage.get_all_indexed_files()
    }

    /// Obtiene un archivo por ID
    pub fn get_file(&self, file_id: &str) -> Result<Option<IndexedFile>> {
        self.storage.get_file_by_id(file_id)
    }

    // ─── Indexación ─────────────────────────

    /// Escanea e indexa todas las carpetas activas
    pub async fn scan_all(&mut self) -> Result<Vec<indexer::pipeline::ScanResult>> {
        self.scan_all_with_progress(None).await
    }

    /// Escanea e indexa todas las carpetas activas con callback de progreso.
    pub async fn scan_all_with_progress(
        &mut self,
        progress_callback: Option<&dyn Fn(IndexProgress)>,
    ) -> Result<Vec<indexer::pipeline::ScanResult>> {
        let folders = self.storage.get_watched_folders()?;
        let mut results = Vec::new();

        for folder in &folders {
            if !folder.active {
                continue;
            }
            let result = self.scan_folder_with_progress(folder, progress_callback).await?;
            results.push(result);
        }

        Ok(results)
    }

    /// Escanea una carpeta específica
    pub async fn scan_folder(
        &mut self,
        folder: &WatchedFolder,
    ) -> Result<indexer::pipeline::ScanResult> {
        self.scan_folder_with_progress(folder, None).await
    }

    /// Escanea una carpeta específica con callback de progreso.
    pub async fn scan_folder_with_progress(
        &mut self,
        folder: &WatchedFolder,
        progress_callback: Option<&dyn Fn(IndexProgress)>,
    ) -> Result<indexer::pipeline::ScanResult> {
        let config = self.config.indexer.clone();
        let mut pipeline =
            IndexPipeline::new(&self.storage, &mut self.vector_store, &self.ollama, config);
        pipeline.scan_folder(folder, progress_callback).await
    }

    /// Procesa embeddings pendientes
    pub async fn process_pending(&mut self) -> Result<u64> {
        let config = self.config.indexer.clone();
        let mut pipeline =
            IndexPipeline::new(&self.storage, &mut self.vector_store, &self.ollama, config);
        pipeline
            .process_pending_embeddings(self.config.indexer.batch_size)
            .await
    }

    /// Regenera todos los embeddings sin re-extraer contenido.
    /// Útil cuando la calidad de búsqueda es mala o se cambió de modelo.
    ///
    /// Proceso:
    /// 1. Limpia todos los vectores del vector store
    /// 2. Resetea el estado de archivos a "content_extracted"
    /// 3. Regenera embeddings en lotes usando el contenido ya extraído
    pub async fn rebuild_embeddings(&mut self) -> Result<u64> {
        tracing::info!("Iniciando rebuild de embeddings...");

        // 1. Limpiar vector store
        self.vector_store.clear()?;

        // 2. Resetear estados en DB
        let reset_count = self.storage.reset_all_to_content_extracted()?;
        tracing::info!("{} archivos marcados para regenerar embeddings", reset_count);

        // 3. Obtener todos los archivos con contenido extraído
        let files = self.storage.get_all_indexed_files()?;
        let total = files.len() as u64;
        let mut processed = 0u64;
        let mut failed = 0u64;

        // Procesar en lotes para mayor eficiencia
        let chunk_size = 8;
        let valid_files: Vec<_> = files.iter().filter(|f| !f.content_full.is_empty()).collect();

        for chunk in valid_files.chunks(chunk_size) {
            let config = self.config.indexer.clone();
            let pipeline =
                IndexPipeline::new(&self.storage, &mut self.vector_store, &self.ollama, config);

            let texts: Vec<String> = chunk
                .iter()
                .map(|f| pipeline.build_embedding_text_pub(f))
                .collect();

            match self.ollama.generate_embeddings(&texts).await {
                Ok(vectors) => {
                    for (i, vector) in vectors.into_iter().enumerate() {
                        let file = chunk[i];
                        if let Err(e) = self.vector_store.insert(&file.id, vector) {
                            tracing::warn!("Error insertando vector para {}: {}", file.filename, e);
                            failed += 1;
                            continue;
                        }
                        self.storage
                            .update_file_status(&file.id, &IndexStatus::Indexed)?;
                        processed += 1;
                    }
                }
                Err(e) => {
                    tracing::warn!("Error en batch de embeddings: {}, intentando uno a uno", e);
                    // Fallback individual
                    for (i, text) in texts.iter().enumerate() {
                        let file = chunk[i];
                        match self.ollama.generate_embedding(text).await {
                            Ok(vector) => {
                                let _ = self.vector_store.insert(&file.id, vector);
                                let _ = self.storage.update_file_status(&file.id, &IndexStatus::Indexed);
                                processed += 1;
                            }
                            Err(_) => { failed += 1; }
                        }
                    }
                }
            }

            if processed % 8 == 0 || processed as usize >= valid_files.len() {
                tracing::info!("Progreso: {}/{} embeddings regenerados", processed, total);
            }
        }

        // 4. Guardar vector store
        self.vector_store.save()?;

        tracing::info!(
            "Rebuild completado: {} regenerados, {} fallidos de {} total",
            processed,
            failed,
            total
        );
        Ok(processed)
    }

    /// Re-indexa completamente: borra todo y re-extrae contenido + embeddings.
    /// Es como empezar de cero pero conservando las carpetas registradas.
    pub async fn reindex_all(&mut self) -> Result<Vec<indexer::pipeline::ScanResult>> {
        tracing::info!("Iniciando re-indexación completa...");

        // 1. Limpiar vector store
        self.vector_store.clear()?;

        // 2. Resetear todos los archivos a pending con hash vacío
        //    Esto fuerza a que scan_folder los re-procese todos
        let reset_count = self.storage.reset_all_to_pending()?;
        tracing::info!("{} archivos reseteados para re-indexación", reset_count);

        // 3. Re-escanear todo
        self.scan_all().await
    }

    /// Re-indexa solo las imágenes (re-extrae con visión + regenera embeddings).
    /// Útil cuando se cambió el prompt de visión sin necesidad de re-procesar texto.
    pub async fn reindex_images(&mut self) -> Result<Vec<indexer::pipeline::ScanResult>> {
        tracing::info!("Iniciando re-indexación de imágenes...");

        // 1. Resetear solo imágenes
        let reset_count = self.storage.reset_images_to_pending()?;
        tracing::info!("{} imágenes reseteadas para re-extracción", reset_count);

        // 2. Limpiar vectores de todo (se regenerarán al escanear)
        self.vector_store.clear()?;

        // 3. Re-escanear — solo imágenes se re-extraerán,
        //    texto/docs se saltarán (hash no cambió)
        let results = self.scan_all().await?;

        // 4. Rebuild embeddings de archivos que perdieron vectores al hacer clear()
        //    (archivos de texto/docs cuyo hash no cambió → scan los saltó)
        let all_files = self.storage.get_all_indexed_files()?;
        let missing: Vec<_> = all_files
            .iter()
            .filter(|f| !f.content_full.is_empty() && !self.vector_store.has_vector(&f.id))
            .collect();

        let total_missing = missing.len();
        let mut rebuilt = 0u64;

        if total_missing > 0 {
            tracing::info!(
                "{} archivos sin vector — regenerando embeddings en lotes",
                total_missing
            );

            let chunk_size = 8;
            for chunk in missing.chunks(chunk_size) {
                let config = self.config.indexer.clone();
                let pipeline = IndexPipeline::new(
                    &self.storage,
                    &mut self.vector_store,
                    &self.ollama,
                    config,
                );

                let texts: Vec<String> = chunk
                    .iter()
                    .map(|f| pipeline.build_embedding_text_pub(f))
                    .collect();

                match self.ollama.generate_embeddings(&texts).await {
                    Ok(vectors) => {
                        for (i, vector) in vectors.into_iter().enumerate() {
                            let file = chunk[i];
                            if let Err(e) = self.vector_store.insert(&file.id, vector) {
                                tracing::warn!("Error insertando vector para {}: {}", file.filename, e);
                                continue;
                            }
                            let _ = self.storage.update_file_status(&file.id, &IndexStatus::Indexed);
                            rebuilt += 1;
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Error batch embeddings: {}, intentando uno a uno", e);
                        for (i, text) in texts.iter().enumerate() {
                            let file = chunk[i];
                            match self.ollama.generate_embedding(text).await {
                                Ok(vector) => {
                                    let _ = self.vector_store.insert(&file.id, vector);
                                    let _ = self.storage.update_file_status(&file.id, &IndexStatus::Indexed);
                                    rebuilt += 1;
                                }
                                Err(_) => {}
                            }
                        }
                    }
                }
            }
        }

        self.vector_store.save()?;
        tracing::info!("{} embeddings reconstruidos de {} faltantes", rebuilt, total_missing);

        Ok(results)
    }

    // ─── Búsqueda ───────────────────────────

    /// Búsqueda semántica de archivos
    pub async fn search(&self, query_text: &str) -> Result<Vec<SearchResult>> {
        let query = SearchQuery::new(query_text);
        let engine = SearchEngine::new(
            &self.storage,
            &self.vector_store,
            &self.ollama,
            self.config.search.clone(),
        );
        engine.search(&query).await
    }

    /// Búsqueda con consulta completa (filtros, límites, etc.)
    pub async fn search_advanced(&self, query: &SearchQuery) -> Result<Vec<SearchResult>> {
        let engine = SearchEngine::new(
            &self.storage,
            &self.vector_store,
            &self.ollama,
            self.config.search.clone(),
        );
        engine.search(query).await
    }

    // ─── Organización virtual ───────────────

    /// Accede al manager del filesystem virtual
    pub fn virtual_fs(&self) -> VirtualFsManager<'_> {
        VirtualFsManager::new(&self.storage, &self.ollama)
    }

    // ─── Estadísticas ───────────────────────

    /// Obtiene estadísticas del sistema
    pub fn stats(&self) -> Result<SystemStats> {
        self.storage.get_stats()
    }

    /// Guarda el estado a disco
    pub fn save(&self) -> Result<()> {
        self.vector_store.save()
    }
}
