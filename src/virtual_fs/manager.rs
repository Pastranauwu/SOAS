use crate::embeddings::OllamaClient;
use crate::error::Result;
use crate::models::*;
use crate::storage::SqliteStorage;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Administrador del sistema de archivos virtual
///
/// Maneja la organización visual de archivos en categorías
/// sin mover ni renombrar los archivos reales.
pub struct VirtualFsManager<'a> {
    storage: &'a SqliteStorage,
    ollama: &'a OllamaClient,
}

/// Representación de árbol de categorías para la UI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryTree {
    pub category: Category,
    pub children: Vec<CategoryTree>,
    pub file_count: usize,
}

impl<'a> VirtualFsManager<'a> {
    pub fn new(storage: &'a SqliteStorage, ollama: &'a OllamaClient) -> Self {
        Self { storage, ollama }
    }

    // ─────────────────────────────────────────
    //  Categorías
    // ─────────────────────────────────────────

    /// Crea una nueva categoría
    pub fn create_category(
        &self,
        name: &str,
        description: &str,
        parent_id: Option<&str>,
    ) -> Result<Category> {
        let mut category = Category::new(name, description);
        category.parent_id = parent_id.map(|s| s.to_string());

        self.storage.create_category(&category)?;
        info!("Categoría creada: {} ({})", name, category.id);

        Ok(category)
    }

    /// Elimina una categoría
    pub fn delete_category(&self, category_id: &str) -> Result<()> {
        self.storage.delete_category(category_id)?;
        info!("Categoría eliminada: {}", category_id);
        Ok(())
    }

    /// Obtiene el árbol completo de categorías
    pub fn get_category_tree(&self) -> Result<Vec<CategoryTree>> {
        let categories = self.storage.get_categories()?;
        let tree = self.build_tree(&categories, None)?;
        Ok(tree)
    }

    fn build_tree(
        &self,
        all_categories: &[Category],
        parent_id: Option<&str>,
    ) -> Result<Vec<CategoryTree>> {
        let mut tree = Vec::new();

        for cat in all_categories {
            let cat_parent = cat.parent_id.as_deref();
            if cat_parent == parent_id {
                let children = self.build_tree(all_categories, Some(&cat.id))?;

                let files = self
                    .storage
                    .get_virtual_files_by_category(&cat.id)?;

                tree.push(CategoryTree {
                    category: cat.clone(),
                    children,
                    file_count: files.len(),
                });
            }
        }

        Ok(tree)
    }

    // ─────────────────────────────────────────
    //  Asignación de archivos
    // ─────────────────────────────────────────

    /// Asigna un archivo a una categoría con nombre virtual
    pub fn assign_file(
        &self,
        file_id: &str,
        category_id: &str,
        virtual_name: &str,
    ) -> Result<VirtualFile> {
        let vf = VirtualFile {
            file_id: file_id.to_string(),
            category_id: category_id.to_string(),
            virtual_name: virtual_name.to_string(),
            notes: None,
            tags: vec![],
            sort_order: 0,
            auto_classified: false,
        };

        self.storage.set_virtual_file(&vf)?;
        debug!(
            "Archivo {} asignado a categoría {} como \"{}\"",
            file_id, category_id, virtual_name
        );

        Ok(vf)
    }

    /// Desasigna un archivo de una categoría
    pub fn unassign_file(&self, file_id: &str, category_id: &str) -> Result<()> {
        self.storage.remove_virtual_file(file_id, category_id)?;
        Ok(())
    }

    /// Actualiza el nombre virtual de un archivo
    pub fn rename_virtual(
        &self,
        file_id: &str,
        category_id: &str,
        new_name: &str,
    ) -> Result<()> {
        // Obtener el virtual file actual
        let vfiles = self.storage.get_virtual_files_for_file(file_id)?;
        if let Some(mut vf) = vfiles.into_iter().find(|v| v.category_id == category_id) {
            vf.virtual_name = new_name.to_string();
            self.storage.set_virtual_file(&vf)?;
        }
        Ok(())
    }

    /// Actualiza las tags de un archivo virtual
    pub fn set_tags(&self, file_id: &str, category_id: &str, tags: Vec<String>) -> Result<()> {
        let vfiles = self.storage.get_virtual_files_for_file(file_id)?;
        if let Some(mut vf) = vfiles.into_iter().find(|v| v.category_id == category_id) {
            vf.tags = tags;
            self.storage.set_virtual_file(&vf)?;
        }
        Ok(())
    }

    /// Actualiza las notas de un archivo virtual
    pub fn set_notes(
        &self,
        file_id: &str,
        category_id: &str,
        notes: Option<String>,
    ) -> Result<()> {
        let vfiles = self.storage.get_virtual_files_for_file(file_id)?;
        if let Some(mut vf) = vfiles.into_iter().find(|v| v.category_id == category_id) {
            vf.notes = notes;
            self.storage.set_virtual_file(&vf)?;
        }
        Ok(())
    }

    /// Lista los archivos de una categoría con sus datos completos
    pub fn list_category_files(
        &self,
        category_id: &str,
    ) -> Result<Vec<(VirtualFile, IndexedFile)>> {
        self.storage.get_virtual_files_by_category(category_id)
    }

    // ─────────────────────────────────────────
    //  Auto-clasificación con LLM
    // ─────────────────────────────────────────

    /// Auto-clasifica un archivo usando LLM
    pub async fn auto_classify_file(&self, file: &IndexedFile) -> Result<Vec<AutoClassification>> {
        let categories = self.storage.get_categories()?;
        let category_names: Vec<String> = categories.iter().map(|c| c.name.clone()).collect();

        // Pedir al LLM que sugiera categorías
        let suggested = self
            .ollama
            .suggest_categories(&file.filename, &file.content_preview, &category_names)
            .await?;

        // Pedir nombre virtual sugerido
        let description = self
            .ollama
            .describe_file(&file.filename, &file.content_preview)
            .await
            .ok();

        let virtual_name = description
            .as_ref()
            .and_then(|d| d.suggested_name.clone())
            .unwrap_or_else(|| file.filename.clone());

        let mut classifications = Vec::new();

        for suggested_cat in &suggested {
            // Buscar si la categoría existe
            let existing = categories
                .iter()
                .find(|c| c.name.eq_ignore_ascii_case(suggested_cat));

            classifications.push(AutoClassification {
                suggested_category: suggested_cat.clone(),
                existing_category_id: existing.map(|c| c.id.clone()),
                suggested_virtual_name: virtual_name.clone(),
                confidence: 0.8, // TODO: obtener confianza real del LLM
            });
        }

        Ok(classifications)
    }

    /// Aplica auto-clasificaciones aprobadas por el usuario
    pub fn apply_classification(
        &self,
        file_id: &str,
        category_id: &str,
        virtual_name: &str,
    ) -> Result<VirtualFile> {
        let vf = VirtualFile {
            file_id: file_id.to_string(),
            category_id: category_id.to_string(),
            virtual_name: virtual_name.to_string(),
            notes: None,
            tags: vec![],
            sort_order: 0,
            auto_classified: true,
        };

        self.storage.set_virtual_file(&vf)?;
        info!(
            "Auto-clasificación aplicada: {} → {} como \"{}\"",
            file_id, category_id, virtual_name
        );

        Ok(vf)
    }

    /// Crea categorías por defecto sugeridas
    pub fn create_default_categories(&self) -> Result<Vec<Category>> {
        let defaults = vec![
            ("Documentos", "Documentos de texto, PDF, Word", "📄", "#4A90D9"),
            ("Imágenes", "Fotos y capturas de pantalla", "🖼️", "#7ED321"),
            ("Hojas de Cálculo", "Archivos CSV, Excel", "📊", "#F5A623"),
            ("Recibos y Facturas", "Comprobantes de pago y facturas", "🧾", "#D0021B"),
            ("Trabajo", "Documentos relacionados con el trabajo", "💼", "#9013FE"),
            ("Personal", "Documentos personales", "👤", "#50E3C2"),
            ("Código", "Archivos de código fuente", "💻", "#BD10E0"),
            ("Configuración", "Archivos de configuración del sistema", "⚙️", "#8B572A"),
        ];

        let mut categories = Vec::new();

        for (name, desc, icon, color) in defaults {
            let mut cat = Category::new(name, desc);
            cat.icon = Some(icon.to_string());
            cat.color = Some(color.to_string());
            self.storage.create_category(&cat)?;
            categories.push(cat);
        }

        info!("Categorías por defecto creadas: {}", categories.len());
        Ok(categories)
    }
}

/// Resultado de auto-clasificación
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoClassification {
    /// Nombre de categoría sugerida
    pub suggested_category: String,
    /// ID de categoría existente que coincide (si existe)
    pub existing_category_id: Option<String>,
    /// Nombre virtual sugerido
    pub suggested_virtual_name: String,
    /// Confianza de la clasificación (0.0 - 1.0)
    pub confidence: f32,
}
