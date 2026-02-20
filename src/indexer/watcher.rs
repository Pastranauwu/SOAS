use crate::error::Result;
use notify::{Config, Event, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::Path;
use std::sync::mpsc;
use tracing::{debug, error, info};

/// Vigila cambios en el filesystem y notifica para re-indexar
pub struct FileWatcher {
    watcher: RecommendedWatcher,
    rx: mpsc::Receiver<notify::Result<Event>>,
}

impl FileWatcher {
    /// Crea un nuevo file watcher
    pub fn new() -> Result<Self> {
        let (tx, rx) = mpsc::channel();

        let watcher = RecommendedWatcher::new(
            move |res| {
                if let Err(e) = tx.send(res) {
                    error!("Error enviando evento de filesystem: {}", e);
                }
            },
            Config::default(),
        )
        .map_err(|e| crate::error::SoasError::Indexer(format!("Error creando watcher: {}", e)))?;

        Ok(Self { watcher, rx })
    }

    /// Comienza a vigilar una carpeta
    pub fn watch(&mut self, path: &Path, recursive: bool) -> Result<()> {
        let mode = if recursive {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };

        self.watcher.watch(path, mode).map_err(|e| {
            crate::error::SoasError::Indexer(format!(
                "Error vigilando {:?}: {}",
                path, e
            ))
        })?;

        info!("Vigilando carpeta: {:?} (recursivo: {})", path, recursive);
        Ok(())
    }

    /// Deja de vigilar una carpeta
    pub fn unwatch(&mut self, path: &Path) -> Result<()> {
        self.watcher.unwatch(path).map_err(|e| {
            crate::error::SoasError::Indexer(format!(
                "Error dejando de vigilar {:?}: {}",
                path, e
            ))
        })?;
        Ok(())
    }

    /// Obtiene los eventos pendientes (no bloqueante)
    pub fn poll_events(&self) -> Vec<FileEvent> {
        let mut events = Vec::new();

        while let Ok(result) = self.rx.try_recv() {
            match result {
                Ok(event) => {
                    let file_event = match event.kind {
                        notify::EventKind::Create(_) => {
                            event.paths.into_iter().map(FileEvent::Created).collect()
                        }
                        notify::EventKind::Modify(_) => {
                            event.paths.into_iter().map(FileEvent::Modified).collect()
                        }
                        notify::EventKind::Remove(_) => {
                            event.paths.into_iter().map(FileEvent::Deleted).collect()
                        }
                        _ => vec![],
                    };
                    events.extend(file_event);
                }
                Err(e) => {
                    error!("Error en evento de filesystem: {}", e);
                }
            }
        }

        if !events.is_empty() {
            debug!("Recibidos {} eventos de filesystem", events.len());
        }

        events
    }
}

/// Tipo de evento de archivo
#[derive(Debug, Clone)]
pub enum FileEvent {
    Created(std::path::PathBuf),
    Modified(std::path::PathBuf),
    Deleted(std::path::PathBuf),
}
