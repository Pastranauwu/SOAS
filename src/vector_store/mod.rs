pub mod memory;

pub use memory::InMemoryVectorStore;

use crate::error::Result;

/// Trait que define la interfaz del vector store
/// Permite intercambiar la implementación (in-memory, Qdrant, etc.)
pub trait VectorStore: Send + Sync {
    /// Inserta un vector asociado a un file_id
    fn insert(&mut self, file_id: &str, vector: Vec<f32>) -> Result<()>;

    /// Elimina un vector por file_id
    fn remove(&mut self, file_id: &str) -> Result<()>;

    /// Elimina todos los vectores
    fn clear(&mut self) -> Result<()>;

    /// Verifica si existe un vector para el file_id dado
    fn has_vector(&self, file_id: &str) -> bool;

    /// Busca los K vectores más similares a la consulta
    fn search(&self, query: &[f32], k: usize, min_score: f32) -> Result<Vec<VectorMatch>>;

    /// Número de vectores almacenados
    fn len(&self) -> usize;

    /// ¿Está vacío?
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Persiste los datos a disco
    fn save(&self) -> Result<()>;

    /// Carga los datos desde disco
    fn load(&mut self) -> Result<()>;
}

/// Resultado de una búsqueda vectorial
#[derive(Debug, Clone)]
pub struct VectorMatch {
    /// ID del archivo asociado
    pub file_id: String,
    /// Puntuación de similitud coseno (0.0 - 1.0)
    pub score: f32,
}
