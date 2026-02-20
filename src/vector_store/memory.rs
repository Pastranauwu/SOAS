use crate::error::{Result, SoasError};
use crate::vector_store::{VectorMatch, VectorStore};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tracing::{debug, info};

/// Vector store en memoria con persistencia a disco
/// Usa búsqueda por fuerza bruta con similitud coseno.
///
/// Funciona bien para colecciones de hasta ~100K archivos.
/// Para colecciones más grandes, se puede reemplazar por Qdrant u otro backend.
pub struct InMemoryVectorStore {
    /// Vectores almacenados: file_id -> vector normalizado
    vectors: HashMap<String, Vec<f32>>,
    /// Ruta del archivo de persistencia
    persist_path: PathBuf,
    /// Dimensión esperada de los vectores
    dimensions: usize,
    /// ¿Hay cambios sin persistir?
    dirty: bool,
}

/// Formato de serialización para persistencia
#[derive(Serialize, Deserialize)]
struct PersistedStore {
    dimensions: usize,
    entries: Vec<PersistedEntry>,
}

#[derive(Serialize, Deserialize)]
struct PersistedEntry {
    file_id: String,
    vector: Vec<f32>,
}

impl InMemoryVectorStore {
    /// Crea un nuevo vector store
    pub fn new(persist_path: PathBuf, dimensions: usize) -> Self {
        Self {
            vectors: HashMap::new(),
            persist_path,
            dimensions,
            dirty: false,
        }
    }

    /// Crea y carga desde disco si existe
    pub fn open(persist_path: PathBuf, dimensions: usize) -> Result<Self> {
        let mut store = Self::new(persist_path, dimensions);
        if store.persist_path.exists() {
            store.load()?;
        }
        Ok(store)
    }

    /// Calcula la similitud coseno entre dos vectores normalizados
    /// Para vectores ya normalizados, es simplemente el producto punto
    #[inline]
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for i in 0..a.len() {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        let denominator = norm_a.sqrt() * norm_b.sqrt();
        if denominator == 0.0 {
            0.0
        } else {
            dot / denominator
        }
    }

    /// Normaliza un vector (L2 norm)
    fn normalize(vector: &mut [f32]) {
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in vector.iter_mut() {
                *x /= norm;
            }
        }
    }
}

impl VectorStore for InMemoryVectorStore {
    fn insert(&mut self, file_id: &str, mut vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(SoasError::VectorStore(format!(
                "Dimensión incorrecta: esperada {}, recibida {}",
                self.dimensions,
                vector.len()
            )));
        }

        Self::normalize(&mut vector);
        self.vectors.insert(file_id.to_string(), vector);
        self.dirty = true;

        debug!(
            "Vector insertado para file_id={}, total={}",
            file_id,
            self.vectors.len()
        );
        Ok(())
    }

    fn remove(&mut self, file_id: &str) -> Result<()> {
        self.vectors.remove(file_id);
        self.dirty = true;
        Ok(())
    }

    fn clear(&mut self) -> Result<()> {
        let count = self.vectors.len();
        self.vectors.clear();
        self.dirty = true;
        info!("Vector store limpiado: {} vectores eliminados", count);
        Ok(())
    }

    fn has_vector(&self, file_id: &str) -> bool {
        self.vectors.contains_key(file_id)
    }

    fn search(&self, query: &[f32], k: usize, min_score: f32) -> Result<Vec<VectorMatch>> {
        if query.len() != self.dimensions {
            return Err(SoasError::VectorStore(format!(
                "Dimensión de consulta incorrecta: esperada {}, recibida {}",
                self.dimensions,
                query.len()
            )));
        }

        // Normalizar la consulta
        let mut normalized_query = query.to_vec();
        Self::normalize(&mut normalized_query);

        // Búsqueda por fuerza bruta (eficiente para <100K vectores)
        let mut scores: Vec<(String, OrderedFloat<f32>)> = self
            .vectors
            .iter()
            .map(|(id, vec)| {
                let score = Self::cosine_similarity(&normalized_query, vec);
                (id.clone(), OrderedFloat(score))
            })
            .filter(|(_, score)| score.0 >= min_score)
            .collect();

        // Ordenar por mayor similitud
        scores.sort_by(|a, b| b.1.cmp(&a.1));

        // Tomar los K mejores
        let results: Vec<VectorMatch> = scores
            .into_iter()
            .take(k)
            .map(|(file_id, score)| VectorMatch {
                file_id,
                score: score.0,
            })
            .collect();

        debug!(
            "Búsqueda vectorial: {} resultados encontrados (de {} vectores)",
            results.len(),
            self.vectors.len()
        );

        Ok(results)
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn save(&self) -> Result<()> {
        if !self.dirty && self.persist_path.exists() {
            return Ok(());
        }

        if let Some(parent) = self.persist_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let store = PersistedStore {
            dimensions: self.dimensions,
            entries: self
                .vectors
                .iter()
                .map(|(file_id, vector)| PersistedEntry {
                    file_id: file_id.clone(),
                    vector: vector.clone(),
                })
                .collect(),
        };

        let encoded = bincode::serialize(&store).map_err(|e| {
            SoasError::VectorStore(format!("Error serializando vector store: {}", e))
        })?;

        std::fs::write(&self.persist_path, encoded)?;

        info!(
            "Vector store guardado: {} vectores en {:?}",
            self.vectors.len(),
            self.persist_path
        );
        Ok(())
    }

    fn load(&mut self) -> Result<()> {
        if !self.persist_path.exists() {
            return Ok(());
        }

        let data = std::fs::read(&self.persist_path)?;
        let store: PersistedStore = bincode::deserialize(&data).map_err(|e| {
            SoasError::VectorStore(format!("Error deserializando vector store: {}", e))
        })?;

        self.dimensions = store.dimensions;
        self.vectors = store
            .entries
            .into_iter()
            .map(|e| (e.file_id, e.vector))
            .collect();

        self.dirty = false;

        info!(
            "Vector store cargado: {} vectores de dimensión {}",
            self.vectors.len(),
            self.dimensions
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_search() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.bin");

        let mut store = InMemoryVectorStore::new(path, 3);

        // Insertar vectores simples
        store
            .insert("file1", vec![1.0, 0.0, 0.0])
            .unwrap();
        store
            .insert("file2", vec![0.0, 1.0, 0.0])
            .unwrap();
        store
            .insert("file3", vec![0.9, 0.1, 0.0])
            .unwrap();

        // Buscar similar a file1
        let results = store.search(&[1.0, 0.0, 0.0], 3, 0.0).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].file_id, "file1"); // Más similar
        assert!((results[0].score - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.bin");

        // Crear y guardar
        {
            let mut store = InMemoryVectorStore::new(path.clone(), 3);
            store.insert("file1", vec![1.0, 0.0, 0.0]).unwrap();
            store.insert("file2", vec![0.0, 1.0, 0.0]).unwrap();
            store.save().unwrap();
        }

        // Cargar y verificar
        {
            let store = InMemoryVectorStore::open(path, 3).unwrap();
            assert_eq!(store.len(), 2);
        }
    }

    #[test]
    fn test_remove() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.bin");

        let mut store = InMemoryVectorStore::new(path, 3);
        store.insert("file1", vec![1.0, 0.0, 0.0]).unwrap();
        assert_eq!(store.len(), 1);

        store.remove("file1").unwrap();
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_min_score_filter() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.bin");

        let mut store = InMemoryVectorStore::new(path, 3);
        store.insert("file1", vec![1.0, 0.0, 0.0]).unwrap();
        store.insert("file2", vec![0.0, 1.0, 0.0]).unwrap();

        // Con min_score alto, solo debería devolver el más similar
        let results = store.search(&[1.0, 0.0, 0.0], 10, 0.9).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file_id, "file1");
    }
}
