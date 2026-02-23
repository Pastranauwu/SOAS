use crate::config::IndexerConfig;
use crate::content;
use crate::embeddings::OllamaClient;
use crate::error::Result;
use crate::models::*;
use crate::storage::SqliteStorage;
use crate::vector_store::VectorStore;
use chrono::Utc;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use tracing::{debug, error, info, warn};
use walkdir::WalkDir;

/// Trunca un string de forma segura respetando límites de caracteres UTF-8
fn safe_truncate(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn safe_slice(s: &str, start: usize, max_bytes: usize) -> &str {
    if s.is_empty() || max_bytes == 0 {
        return "";
    }

    let mut begin = start.min(s.len());
    while begin < s.len() && !s.is_char_boundary(begin) {
        begin += 1;
    }

    if begin >= s.len() {
        return "";
    }

    let mut end = (begin + max_bytes).min(s.len());
    while end > begin && !s.is_char_boundary(end) {
        end -= 1;
    }

    &s[begin..end]
}

/// Pipeline de indexación: descubre, extrae, embebe y almacena archivos
pub struct IndexPipeline<'a> {
    storage: &'a SqliteStorage,
    vector_store: &'a mut dyn VectorStore,
    ollama: &'a OllamaClient,
    config: IndexerConfig,
}

impl<'a> IndexPipeline<'a> {
    const MIN_LLM_CHARS: usize = 600;

    fn is_image_extension(ext: &str) -> bool {
        matches!(
            ext,
            "jpg" | "jpeg" | "png" | "bmp" | "tiff" | "tif" | "webp" | "gif" | "svg" | "heic" | "heif"
        )
    }

    fn normalize_filename_for_text(filename: &str) -> String {
        filename
            .replace(['_', '-', '.'], " ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn derive_keywords(text: &str, max_keywords: usize) -> Vec<String> {
        let stopwords = [
            "de", "del", "la", "las", "el", "los", "y", "o", "en", "con", "para", "por", "un", "una",
            "uno", "una", "que", "es", "al", "se", "sin", "archivo", "imagen", "resumen", "visual",
            "muestra", "puede", "ver", "tiene", "como", "son", "esta", "este", "esto", "una",
            "hay", "con", "muy", "mas", "pero", "también", "fue", "ser", "sobre", "todo",
            "posible", "contenido", "nombre", "semántico", "semantic",
        ];

        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for token in text
            .split(|c: char| !c.is_alphanumeric())
            .map(|w| w.trim().to_lowercase())
            .filter(|w| w.len() >= 3)
        {
            if stopwords.contains(&token.as_str()) {
                continue;
            }
            if seen.insert(token.clone()) {
                out.push(token);
            }
            if out.len() >= max_keywords {
                break;
            }
        }

        out
    }

    /// Extrae keywords semánticas ricas para imágenes.
    ///
    /// A diferencia de `derive_keywords` (que tokeniza palabras sueltas),
    /// esta función:
    /// 1. Detecta entidades y frases significativas del nombre de archivo
    /// 2. Extrae conceptos clave de la descripción visual (si existe)
    /// 3. Produce keywords que son útiles para búsqueda en lenguaje natural
    ///
    /// Ejemplo: "ine_posterior.jpeg" + descripción de INE →
    /// ["INE", "credencial", "identificación", "INE posterior", "documento oficial"]
    /// en vez de: ["ine", "posterior", "jpeg"]
    fn derive_image_keywords(filename: &str, vision_description: &str) -> Vec<String> {
        let mut keywords = Vec::new();
        let mut seen = std::collections::HashSet::new();

        let clean_name = filename
            .replace(['_', '-', '.'], " ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();

        let desc_lower = vision_description.to_lowercase();
        let combined = format!("{} {}", clean_name, desc_lower);

        // ── 1) Detectar entidades/conceptos conocidos ──────────────────────
        // Mapeo: (patrones_a_buscar, keywords_a_generar)
        let entity_map: &[(&[&str], &[&str])] = &[
            // Documentos de identidad
            (&["ine", "instituto nacional electoral", "credencial para votar", "credencial elector"],
             &["INE", "credencial", "identificación oficial", "credencial de elector"]),
            (&["curp"], &["CURP", "clave única de registro"]),
            (&["rfc"], &["RFC", "registro federal de contribuyentes"]),
            (&["pasaporte"], &["pasaporte", "documento de viaje"]),
            (&["licencia", "conducir"], &["licencia de conducir"]),
            (&["acta", "nacimiento"], &["acta de nacimiento"]),
            (&["acta", "matrimonio"], &["acta de matrimonio"]),
            (&["comprobante", "domicilio"], &["comprobante de domicilio"]),
            // Documentos laborales/oficiales
            (&["carta", "hechos"], &["carta de hechos"]),
            (&["informe", "actividades"], &["informe de actividades"]),
            (&["acta administrativa"], &["acta administrativa"]),
            (&["licitacion", "licitación"], &["licitación"]),
            (&["constancia"], &["constancia"]),
            (&["diploma", "certificado", "titulo", "título"], &["certificado", "diploma"]),
            (&["factura"], &["factura"]),
            (&["recibo", "comprobante"], &["recibo", "comprobante"]),
            (&["contrato"], &["contrato"]),
            (&["nómina", "nomina"], &["nómina", "recibo de nómina"]),
            // Orientación de imagen
            (&["frente", "frontal", "front"], &["frente", "vista frontal"]),
            (&["posterior", "reverso", "atras", "back"], &["reverso", "vista posterior"]),
            // Fotos
            (&["selfie", "retrato", "foto"], &["fotografía"]),
            (&["screenshot", "captura", "pantalla"], &["captura de pantalla"]),
        ];

        for (patterns, kw_to_add) in entity_map {
            let matched = patterns.iter().all(|p| combined.contains(p));
            if matched {
                for kw in *kw_to_add {
                    let kw_lower = kw.to_lowercase();
                    if seen.insert(kw_lower) {
                        keywords.push(kw.to_string());
                    }
                }
            }
        }

        // ── 2) Extraer palabras significativas del nombre ──────────────────
        let name_stopwords = [
            "jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp", "svg", "heic",
            "img", "image", "foto", "photo", "pic", "screenshot",
        ];
        for word in clean_name.split_whitespace() {
            if word.len() >= 3
                && !name_stopwords.contains(&word)
                && seen.insert(word.to_string())
            {
                keywords.push(word.to_string());
            }
        }

        // ── 3) Extraer conceptos sustantivos de la descripción visual ──────
        // Buscar frases sustantivas relevantes (no solo tokens sueltos)
        if !vision_description.is_empty() {
            let desc_stopwords = [
                "imagen", "muestra", "puede", "tiene", "como", "donde", "esta",
                "este", "esto", "sobre", "todo", "también", "sido", "siendo",
                "hay", "ver", "con", "sin", "los", "las", "del", "una", "uno",
                "para", "por", "que", "son", "fue", "ser", "más", "muy",
                "esquina", "superior", "inferior", "derecha", "izquierda",
                "parte", "lado", "fondo", "centro", "arriba", "abajo",
                "chars", "oficial",
            ];

            // Extraer palabras relevantes de la descripción
            for token in desc_lower
                .split(|c: char| !c.is_alphanumeric() && c != 'á' && c != 'é' && c != 'í' && c != 'ó' && c != 'ú' && c != 'ñ')
                .map(|w| w.trim())
                .filter(|w| w.len() >= 4)
            {
                if !desc_stopwords.contains(&token) && seen.insert(token.to_string()) {
                    keywords.push(token.to_string());
                    if keywords.len() >= 12 {
                        break;
                    }
                }
            }
        }

        // Limitar a un máximo razonable
        keywords.truncate(10);
        keywords
    }

    fn should_use_llm_enrichment_for_file(file: &IndexedFile) -> bool {
        if Self::is_image_extension(&file.extension) {
            return false;
        }

        if file.content_preview.trim().is_empty() {
            return false;
        }

        let content_len = file.content_full.len().max(file.content_preview.len());

        if content_len >= 2200 {
            return true;
        }

        let ext = file.extension.as_str();
        if matches!(ext, "pdf" | "docx" | "doc" | "pptx" | "ppt" | "xlsx" | "xls")
            && content_len >= Self::MIN_LLM_CHARS
        {
            return true;
        }

        let fname = file.filename.to_lowercase();
        let relevance_hints = [
            "informe",
            "acta",
            "contrato",
            "licit",
            "carta",
            "guia",
            "formato",
            "reporte",
            "oficio",
            "acuerdo",
        ];

        relevance_hints.iter().any(|h| fname.contains(h)) && content_len >= Self::MIN_LLM_CHARS
    }

    fn ensure_local_metadata_baseline(file: &mut IndexedFile) {
        if file.metadata.title.is_none() {
            file.metadata.title = Some(Self::normalize_filename_for_text(&file.filename));
        }

        if file.metadata.description.is_none() {
            let fallback_desc = if file.content_preview.is_empty() {
                format!("{}", Self::normalize_filename_for_text(&file.filename))
            } else {
                safe_truncate(&file.content_preview, 220).to_string()
            };
            file.metadata.description = Some(fallback_desc);
        }

        if file.metadata.keywords.is_empty() {
            let mut source = Self::normalize_filename_for_text(&file.filename);
            if let Some(desc) = &file.metadata.description {
                source.push(' ');
                source.push_str(desc);
            }
            file.metadata.keywords = Self::derive_keywords(&source, 8);
        }

        if file.metadata.semantic_tags.is_empty() {
            file.metadata.semantic_tags = file.metadata.keywords.iter().take(4).cloned().collect();
        }
    }

    fn build_llm_context_excerpt(content: &str) -> String {
        if content.is_empty() {
            return String::new();
        }

        let total_len = content.len();
        if total_len <= 1800 {
            return safe_truncate(content, 1800).to_string();
        }

        let chunk = 520;
        let head = safe_slice(content, 0, chunk);
        let mid_start = total_len.saturating_div(2).saturating_sub(chunk / 2);
        let middle = safe_slice(content, mid_start, chunk);
        let tail_start = total_len.saturating_sub(chunk);
        let tail = safe_slice(content, tail_start, chunk);

        format!(
            "[INICIO]\n{}\n\n[MEDIO]\n{}\n\n[FINAL]\n{}",
            head, middle, tail
        )
    }

    pub fn new(
        storage: &'a SqliteStorage,
        vector_store: &'a mut dyn VectorStore,
        ollama: &'a OllamaClient,
        config: IndexerConfig,
    ) -> Self {
        Self {
            storage,
            vector_store,
            ollama,
            config,
        }
    }

    /// Escanea una carpeta y registra los archivos nuevos/modificados
    pub async fn scan_folder(
        &mut self,
        folder: &WatchedFolder,
        progress_callback: Option<&dyn Fn(IndexProgress)>,
    ) -> Result<ScanResult> {
        info!("Escaneando carpeta: {:?}", folder.path);

        let mut result = ScanResult {
            new_files: 0,
            updated_files: 0,
            deleted_files: 0,
            failed_files: 0,
            total_scanned: 0,
        };

        // Obtener archivos existentes en la DB para esta carpeta
        let existing = self.storage.get_all_file_paths_and_hashes()?;
        let existing_paths: std::collections::HashMap<String, (String, String)> = existing
            .into_iter()
            .filter(|(_, path, _)| path.starts_with(&folder.path.to_string_lossy().to_string()))
            .map(|(id, path, hash)| (path, (id, hash)))
            .collect();

        let mut seen_paths = std::collections::HashSet::new();

        // Recorrer el directorio
        let walker = if folder.recursive {
            WalkDir::new(&folder.path)
        } else {
            WalkDir::new(&folder.path).max_depth(1)
        };

        let mut entries: Vec<PathBuf> = walker
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .filter(|e| self.should_index(e.path()))
            .map(|e| e.path().to_path_buf())
            .collect();

        // ── Ordenar: NO-imágenes primero, imágenes al final ──────────────
        // Ollama solo ejecuta un modelo a la vez en RAM. Intercalar tipos
        // (imagen → texto → imagen) fuerza swap de modelo cada vez (~10-15s
        // de carga en CPU). Agrupar por tipo minimiza los swaps a solo 1.
        entries.sort_by_key(|p| {
            let ext = p.extension()
                .map(|e| e.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            if Self::is_image_extension(&ext) { 1 } else { 0 }
        });

        let total = entries.len() as u64;
        let image_count = entries.iter().filter(|p| {
            let ext = p.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
            Self::is_image_extension(&ext)
        }).count();
        info!("Encontrados {} archivos para procesar ({} documentos, {} imágenes)",
              total, total as usize - image_count, image_count);

        for (idx, path) in entries.iter().enumerate() {
            let path_str = path.to_string_lossy().to_string();
            seen_paths.insert(path_str.clone());
            result.total_scanned += 1;

            // Reportar progreso
            if let Some(cb) = progress_callback {
                cb(IndexProgress {
                    total,
                    processed: idx as u64,
                    failed: result.failed_files,
                    current_file: Some(path_str.clone()),
                    phase: IndexPhase::Scanning,
                });
            }

            // Verificar si ya existe y calcular hash
            let file_hash = match self.compute_hash(path) {
                Ok(h) => h,
                Err(e) => {
                    warn!("Error calculando hash de {:?}: {}", path, e);
                    result.failed_files += 1;
                    continue;
                }
            };

            if let Some((file_id, existing_hash)) = existing_paths.get(&path_str) {
                if *existing_hash == file_hash {
                    // Archivo sin cambios, skip
                    continue;
                }
                // Archivo modificado, actualizar
                debug!("Archivo modificado: {:?}", path);
                match self.process_file(path, Some(file_id)).await {
                    Ok(_) => result.updated_files += 1,
                    Err(e) => {
                        error!("Error procesando {:?}: {}", path, e);
                        result.failed_files += 1;
                    }
                }
            } else {
                // Archivo nuevo
                debug!("Archivo nuevo: {:?}", path);
                match self.process_file(path, None).await {
                    Ok(_) => result.new_files += 1,
                    Err(e) => {
                        error!("Error procesando {:?}: {}", path, e);
                        result.failed_files += 1;
                    }
                }
            }
        }

        // Detectar archivos eliminados
        for (path, (file_id, _)) in &existing_paths {
            if !seen_paths.contains(path) {
                debug!("Archivo eliminado: {}", path);
                if let Err(e) = self.storage.delete_file(file_id) {
                    warn!("Error eliminando archivo de DB: {}", e);
                } else {
                    let _ = self.vector_store.remove(file_id);
                    result.deleted_files += 1;
                }
            }
        }

        // Actualizar timestamp de último scan
        self.storage.update_folder_last_scan(&folder.id)?;

        // Persistir vector store
        self.vector_store.save()?;

        info!(
            "Escaneo completado: {} nuevos, {} actualizados, {} eliminados, {} errores",
            result.new_files, result.updated_files, result.deleted_files, result.failed_files
        );

        Ok(result)
    }

    /// Procesa un archivo individual: extrae contenido, genera embedding, almacena
    pub async fn process_file(
        &mut self,
        path: &Path,
        existing_id: Option<&str>,
    ) -> Result<IndexedFile> {
        let metadata = std::fs::metadata(path)?;

        // Crear o actualizar el registro del archivo
        let mut file = if let Some(id) = existing_id {
            self.storage
                .get_file_by_id(id)?
                .unwrap_or_else(|| IndexedFile::new(path.to_path_buf()))
        } else {
            IndexedFile::new(path.to_path_buf())
        };

        file.path = path.to_path_buf();
        file.filename = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        file.extension = path
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        file.size = metadata.len();
        file.mime_type = mime_guess::from_path(path)
            .first_or_octet_stream()
            .to_string();
        file.content_hash = self.compute_hash(path)?;
        file.indexed_at = Utc::now();

        // Extraer fechas del filesystem
        if let Ok(created) = metadata.created() {
            file.created_at = chrono::DateTime::from(created);
        }
        if let Ok(modified) = metadata.modified() {
            file.modified_at = chrono::DateTime::from(modified);
        }

        // Verificar tamaño máximo
        if file.size > self.config.max_file_size {
            file.index_status = IndexStatus::Failed("Archivo demasiado grande".to_string());
            self.storage.upsert_file(&file)?;
            return Ok(file);
        }

        // Paso 1: Extraer contenido
        file.index_status = IndexStatus::Pending;
        match content::extract_content(path, Some(self.ollama)).await {
            Ok(extracted) => {
                file.content_full = if extracted.text.len() > self.config.max_content_length {
                    extracted
                        .truncated_text(self.config.max_content_length)
                        .to_string()
                } else {
                    extracted.text.clone()
                };
                file.content_preview = extracted
                    .truncated_text(self.config.preview_length)
                    .to_string();

                // Actualizar metadatos
                if let Some(title) = extracted.title {
                    file.metadata.title = Some(title);
                }
                if let Some(author) = extracted.author {
                    file.metadata.author = Some(author);
                }
                if let Some(pages) = extracted.page_count {
                    file.metadata.page_count = Some(pages);
                }
                for (k, v) in extracted.extra {
                    file.metadata.extra.insert(k, v);
                }

                file.index_status = IndexStatus::ContentExtracted;
                info!(
                    "📄 Contenido [{:?}]: {} chars | preview: {:?}",
                    path.file_name().unwrap_or_default(),
                    file.content_full.len(),
                    safe_truncate(&file.content_preview, 250)
                );
            }
            Err(e) => {
                warn!("Error extrayendo contenido de {:?}: {}", path, e);
                file.index_status =
                    IndexStatus::Failed(format!("Error de extracción: {}", e));
                self.storage.upsert_file(&file)?;
                return Ok(file);
            }
        }

        let is_image = Self::is_image_extension(&file.extension);

        // Optimización CPU + calidad para imágenes:
        // - No usar describe_file (llamada LLM extra y lenta)
        // - Evitar alucinaciones cuando el resumen visual vino de fallback
        // - Derivar metadatos localmente desde resumen visual + nombre de archivo
        if is_image {
            file.metadata.title = Some(Self::normalize_filename_for_text(&file.filename));
            file.metadata.description = if file.content_preview.is_empty() {
                Some(Self::normalize_filename_for_text(&file.filename))
            } else {
                Some(file.content_preview.clone())
            };

            // Usar extracción semántica de keywords que entiende contexto,
            // no solo tokenización de palabras sueltas.
            // Ej: "ine_posterior.jpeg" + desc "documento INE" →
            //   ["INE", "credencial", "identificación oficial", "posterior"]
            // en vez de: ["ine", "posterior", "jpeg"]
            file.metadata.keywords = Self::derive_image_keywords(
                &file.filename,
                &file.content_preview,
            );

            let mut tags = vec!["imagen".to_string()];
            tags.extend(file.metadata.keywords.iter().take(3).cloned());
            file.metadata.semantic_tags = tags;
            file.metadata.content_type_group = Some("imagen".to_string());

            info!(
                "🏷️  Metadatos imagen [{:?}]: keywords={:?}, desc_len={}",
                path.file_name().unwrap_or_default(),
                &file.metadata.keywords,
                file.metadata.description.as_ref().map(|d| d.len()).unwrap_or(0),
            );
        } else {
            // Siempre dejar baseline local para robustez y velocidad.
            Self::ensure_local_metadata_baseline(&mut file);
        }

        let image_fallback_method = file
            .metadata
            .extra
            .get("ocr")
            .cloned()
            .unwrap_or_default();
        let should_rescue_image_with_llm = is_image
            && self.config.use_llm_enrichment
            && image_fallback_method.starts_with("qwen3vl-fallback");

        // Paso 2: (Opcional) Enriquecer metadatos con LLM para no-imágenes prioritarias,
        // o rescatar imágenes que cayeron en fallback de visión.
        let should_use_llm = self.config.use_llm_enrichment
            && !is_image
            && Self::should_use_llm_enrichment_for_file(&file);

        if should_use_llm || should_rescue_image_with_llm {
            let llm_context = if !file.content_full.is_empty() {
                Self::build_llm_context_excerpt(&file.content_full)
            } else {
                safe_truncate(&file.content_preview, 1200).to_string()
            };

            // Extraer ruta de carpeta relativa al home del usuario para contexto
            let folder_context = file.path.parent()
                .map(|p| {
                    let ps = p.to_string_lossy();
                    // Recortar /home/usuario/ para mostrar solo la ruta relativa
                    if let Some(idx) = ps.find("/home/") {
                        let rest = &ps[idx + 6..];
                        if let Some(slash) = rest.find('/') {
                            return rest[slash + 1..].to_string();
                        }
                    }
                    ps.to_string()
                })
                .unwrap_or_default();

            match self.ollama.describe_file(&file.filename, &llm_context, &folder_context).await {
                Ok(desc) => {
                    // Para imágenes en fallback: preferir metadatos del LLM si son útiles.
                    // Para no-imágenes: mantener estrategia conservadora (rellenar vacíos).
                    if should_rescue_image_with_llm {
                        if !desc.title.trim().is_empty() {
                            file.metadata.title = Some(desc.title);
                        }
                        if !desc.description.trim().is_empty() {
                            file.metadata.description = Some(desc.description);
                        }
                        if !desc.keywords.is_empty() {
                            file.metadata.keywords = desc.keywords;
                        }
                        if !desc.semantic_tags.is_empty() {
                            file.metadata.semantic_tags = desc.semantic_tags;
                        }
                        if let Some(lang) = desc.language {
                            file.metadata.language = Some(lang);
                        }
                        file.metadata.content_type_group = Some(
                            desc.content_type_group.unwrap_or_else(|| "imagen".to_string())
                        );
                        info!(
                            "🧠 Rescate LLM imagen fallback [{:?}] (método={}): keywords={:?}, tags={:?}",
                            path.file_name().unwrap_or_default(),
                            image_fallback_method,
                            &file.metadata.keywords,
                            &file.metadata.semantic_tags
                        );
                    } else {
                        // Solo sobreescribir si no había metadatos previos
                        if file.metadata.title.is_none() {
                            file.metadata.title = Some(desc.title);
                        }
                        if file.metadata.description.is_none() {
                            file.metadata.description = Some(desc.description);
                        }
                        if file.metadata.keywords.is_empty() {
                            file.metadata.keywords = desc.keywords;
                        }
                        if file.metadata.semantic_tags.is_empty() {
                            file.metadata.semantic_tags = desc.semantic_tags;
                        }
                        if file.metadata.language.is_none() {
                            file.metadata.language = desc.language;
                        }
                        if file.metadata.content_type_group.is_none() {
                            file.metadata.content_type_group = desc.content_type_group;
                        }
                        info!(
                            "🏷️  LLM enriquecimiento [{:?}]: keywords={:?}, tags={:?}",
                            path.file_name().unwrap_or_default(),
                            &file.metadata.keywords,
                            &file.metadata.semantic_tags
                        );
                    }

                    if let Some(ref desc) = file.metadata.description {
                        info!("   desc: {:?}", safe_truncate(desc, 100));
                    }
                }
                Err(e) => {
                    warn!("No se pudo enriquecer metadatos de {:?}: {}", path, e);
                }
            }
        } else if self.config.use_llm_enrichment && !is_image {
            info!(
                "⏭️  LLM omitido por prioridad [{:?}] (len={} chars)",
                path.file_name().unwrap_or_default(),
                file.content_full.len().max(file.content_preview.len())
            );
        }

        // Si no se enriqueció con LLM, detectar el grupo de tipo por extensión/mime
        if file.metadata.content_type_group.is_none() {
            file.metadata.content_type_group = Some(Self::detect_content_type_group(
                &file.extension,
                &file.mime_type,
            ));
        }

        // Paso 3: Generar embedding
        if !file.content_full.is_empty() {
            // Construir texto para embedding: combina nombre, contenido y metadatos
            let embedding_text = self.build_embedding_text(&file);
            debug!(
                "Embedding text [{:?}] ({} chars): {:?}",
                path.file_name().unwrap_or_default(),
                embedding_text.len(),
                safe_truncate(&embedding_text, 300)
            );

            match self.ollama.generate_embedding(&embedding_text).await {
                Ok(vector) => {
                    if let Err(e) = self.vector_store.insert(&file.id, vector) {
                        warn!("Error insertando vector para {:?}: {}", path, e);
                    }
                    file.index_status = IndexStatus::Indexed;
                    info!("✅ Embedding generado [{:?}]", path.file_name().unwrap_or_default());
                }
                Err(e) => {
                    warn!("Error generando embedding para {:?}: {}", path, e);
                    // Si ya estaba indexed (re-scan), NO degradar el status.
                    // El vector anterior sigue en el store y funciona.
                    if existing_id.is_some() {
                        file.index_status = IndexStatus::Indexed;
                    }
                    // Si es nuevo, se queda como content_extracted para reintentar
                }
            }
        }

        // Guardar en SQLite
        self.storage.upsert_file(&file)?;

        Ok(file)
    }

    /// Procesa archivos pendientes de embedding en lotes para mayor eficiencia.
    /// Ollama /api/embed soporta múltiples textos en una sola petición.
    pub async fn process_pending_embeddings(&mut self, batch_size: usize) -> Result<u64> {
        let pending = self
            .storage
            .get_files_by_status("content_extracted", batch_size)?;

        if pending.is_empty() {
            return Ok(0);
        }

        // Preparar textos y IDs (excluir archivos sin contenido)
        let mut texts = Vec::new();
        let mut file_ids = Vec::new();
        for file in &pending {
            if file.content_full.is_empty() {
                continue;
            }
            texts.push(self.build_embedding_text(file));
            file_ids.push(file.id.clone());
        }

        if texts.is_empty() {
            return Ok(0);
        }

        let mut processed = 0u64;
        let chunk_size = 8; // Procesar en lotes de 8 para no saturar Ollama

        for chunk_start in (0..texts.len()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(texts.len());
            let text_batch = &texts[chunk_start..chunk_end];
            let id_batch = &file_ids[chunk_start..chunk_end];

            match self.ollama.generate_embeddings(text_batch).await {
                Ok(vectors) => {
                    for (i, vector) in vectors.into_iter().enumerate() {
                        let fid = &id_batch[i];
                        if let Err(e) = self.vector_store.insert(fid, vector) {
                            warn!("Error insertando vector: {}", e);
                            continue;
                        }
                        self.storage
                            .update_file_status(fid, &IndexStatus::Indexed)?;
                        processed += 1;
                    }
                }
                Err(e) => {
                    warn!("Error generando batch de embeddings: {}", e);
                    // Fallback: intentar uno por uno
                    for (i, text) in text_batch.iter().enumerate() {
                        let fid = &id_batch[i];
                        match self.ollama.generate_embedding(text).await {
                            Ok(vector) => {
                                let _ = self.vector_store.insert(fid, vector);
                                let _ = self.storage.update_file_status(fid, &IndexStatus::Indexed);
                                processed += 1;
                            }
                            Err(e2) => {
                                warn!("Error individual embedding para {}: {}", fid, e2);
                            }
                        }
                    }
                }
            }
        }

        if processed > 0 {
            self.vector_store.save()?;
            info!("{} embeddings pendientes procesados", processed);
        }

        Ok(processed)
    }

    /// Construye el texto optimizado para generar el embedding
    fn build_embedding_text(&self, file: &IndexedFile) -> String {
        Self::make_embedding_text(file)
    }

    /// Versión pública estática para usar desde fuera del pipeline
    pub fn build_embedding_text_pub(&self, file: &IndexedFile) -> String {
        Self::make_embedding_text(file)
    }

    /// Construye el texto optimizado para generar el embedding.
    ///
    /// Prioriza un resumen semántico del contenido (metadata.description + extracto)
    /// para capturar mejor intención de búsqueda en lenguaje natural.
    /// Las keywords se guardan como metadata en DB y NO se inyectan al embedding,
    /// para evitar ruido léxico y sobreajuste por términos sueltos.
    fn make_embedding_text(file: &IndexedFile) -> String {
        let mut parts = Vec::new();

        // ═══════════════════════════════════════════════════════════════════════
        // PRIORIDAD: el contenido específico va PRIMERO para que el modelo de
        // embedding lo pondere más. La info genérica (tipo, fecha) va al final.
        // ═══════════════════════════════════════════════════════════════════════

        let is_image = matches!(
            file.extension.as_str(),
            "jpg" | "jpeg" | "png" | "bmp" | "tiff" | "webp" | "gif" | "svg" | "heic"
        );

        // Nombre limpio del archivo (reutilizado abajo)
        let clean_name = file.filename
            .replace(['_', '-', '.'], " ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");

        // ── Para imágenes: el nombre del archivo es la señal MÁS importante ──
        // Sin OCR, el nombre "ine_frente.jpeg" es lo único que identifica el
        // contenido. Va PRIMERO para máximo peso en el espacio de embeddings.
        if is_image {
            parts.push(format!("Archivo: {} ({})", file.filename, clean_name));

            // Inyectar keywords semánticas para imágenes.
            // Ahora son frases/conceptos reales ("credencial INE", "identificación oficial")
            // no tokens sueltos ("ine", "posterior", "jpeg"), así que SÍ aportan al embedding.
            if !file.metadata.keywords.is_empty() {
                parts.push(format!("Conceptos: {}", file.metadata.keywords.join(", ")));
            }
        }

        // ── 0) Contexto de carpeta ─────────────────────────────────────────
        // La ruta de carpeta aporta contexto semántico valioso:
        // "Descargas/trabajo/" vs "Documentos/escuela/" ayuda a diferenciar.
        if let Some(parent) = file.path.parent() {
            let folder_str = parent.to_string_lossy();
            // Extraer ruta relativa (sin /home/usuario/)
            let relative = if let Some(idx) = folder_str.find("/home/") {
                let rest = &folder_str[idx + 6..];
                rest.find('/').map(|s| &rest[s + 1..]).unwrap_or("")
            } else {
                &folder_str
            };
            if !relative.is_empty() {
                parts.push(format!("Carpeta: {}", relative));
            }
        }

        // ── 1) Resumen semántico (señal principal para embedding) ───────────
        if let Some(ref desc) = file.metadata.description {
            if !desc.trim().is_empty() {
                parts.push(format!("Resumen semántico: {}", desc.trim()));
            }
        }

        if let Some(ref title) = file.metadata.title {
            if title != &file.filename {
                parts.push(format!("Título: {}", title));
            }
        }

        // ── 2) Meta-resumen del contenido extraído ────────────────────────────
        // Más corto para CPU y más semántico para búsqueda.
        let content_limit = if is_image { 600 } else { 1000 };
        if !file.content_full.is_empty() {
            if !is_image && file.content_full.len() > 2600 {
                let chunk = 320;
                let head = safe_slice(&file.content_full, 0, chunk);
                let mid_start = file.content_full.len().saturating_div(2).saturating_sub(chunk / 2);
                let middle = safe_slice(&file.content_full, mid_start, chunk);
                let tail_start = file.content_full.len().saturating_sub(chunk);
                let tail = safe_slice(&file.content_full, tail_start, chunk);

                parts.push(format!(
                    "Extracto de contenido: [inicio] {} [medio] {} [final] {}",
                    head, middle, tail
                ));
            } else {
                let content = safe_truncate(&file.content_full, content_limit);
                parts.push(format!("Extracto de contenido: {}", content));
            }
        } else if !file.content_preview.is_empty() {
            let preview = safe_truncate(&file.content_preview, 500);
            parts.push(format!("Extracto de contenido: {}", preview));
        }

        // ── 3) Nombre del archivo (señal moderada, ya incluido arriba para imágenes)
        if !is_image {
            parts.push(format!("Archivo: {} ({})", file.filename, clean_name));
        }

        if let Some(ref author) = file.metadata.author {
            parts.push(format!("Autor: {}", author));
        }

        // ── 4) Info genérica (menor peso, va al final) ────────────────────────
        // Tipo de archivo: etiqueta SIMPLE sin sinónimos
        // "imagen" no "imagen fotografía foto" — los sinónimos crean ruido
        let type_label = file.metadata.content_type_group
            .clone()
            .unwrap_or_else(|| Self::detect_content_type_group(&file.extension, &file.mime_type));
        parts.push(format!("Tipo: {}", type_label));

        // Contexto temporal
        let year = file.modified_at.format("%Y").to_string();
        let month_es = match file.modified_at.format("%m").to_string().as_str() {
            "01" => "enero", "02" => "febrero", "03" => "marzo",
            "04" => "abril", "05" => "mayo", "06" => "junio",
            "07" => "julio", "08" => "agosto", "09" => "septiembre",
            "10" => "octubre", "11" => "noviembre", "12" => "diciembre",
            _ => "",
        };
        if !month_es.is_empty() {
            parts.push(format!("Fecha: {} {}", month_es, year));
        }

        if let Some(ref lang) = file.metadata.language {
            parts.push(format!("Idioma: {}", lang));
        }

        if let Some(pages) = file.metadata.page_count {
            parts.push(format!("Páginas: {}", pages));
        }

        parts.join("\n")
    }

    /// Determina el grupo semántico de tipo de contenido a partir de extensión/MIME.
    /// Devuelve una etiqueta legible en español para incluir en embeddings.
    pub fn detect_content_type_group(extension: &str, mime_type: &str) -> String {
        let ext = extension.to_lowercase();
        let mime = mime_type.to_lowercase();

        // Etiquetas SIMPLES — sin sinónimos. Los sinónimos crean ruido en
        // el espacio de embeddings (ej: "imagen fotografía foto" hace que
        // TODAS las imágenes matcheen queries con "foto" igualmente).
        // La diferenciación real viene del contenido extraído, no de la etiqueta.

        if matches!(
            ext.as_str(),
            "jpg" | "jpeg" | "png" | "gif" | "bmp" | "tiff" | "tif"
            | "webp" | "svg" | "ico" | "heic" | "heif" | "raw"
        ) || mime.starts_with("image/") {
            return "imagen".to_string();
        }

        if matches!(ext.as_str(), "mp4" | "avi" | "mkv" | "mov" | "wmv" | "flv" | "webm")
            || mime.starts_with("video/") {
            return "video".to_string();
        }

        if matches!(ext.as_str(), "mp3" | "wav" | "flac" | "ogg" | "m4a" | "aac" | "wma")
            || mime.starts_with("audio/") {
            return "audio".to_string();
        }

        if matches!(ext.as_str(), "pdf") {
            return "documento PDF".to_string();
        }

        if matches!(ext.as_str(), "docx" | "doc" | "odt" | "rtf") {
            return "documento de texto".to_string();
        }

        if matches!(ext.as_str(), "txt" | "md" | "log") {
            return "texto plano".to_string();
        }

        if matches!(ext.as_str(), "xlsx" | "xls" | "csv" | "tsv" | "ods") {
            return "hoja de cálculo".to_string();
        }

        if matches!(ext.as_str(), "pptx" | "ppt" | "odp") {
            return "presentación".to_string();
        }

        if matches!(
            ext.as_str(),
            "py" | "js" | "ts" | "rs" | "go" | "java" | "cpp" | "c"
            | "h" | "sh" | "rb" | "php" | "swift" | "kt" | "html" | "css"
        ) || mime.contains("text/x-")
            || mime.contains("application/x-sh") {
            return "código fuente".to_string();
        }

        if matches!(ext.as_str(), "zip" | "rar" | "tar" | "gz" | "7z" | "bz2")
            || mime.contains("zip")
            || mime.contains("compressed") {
            return "archivo comprimido".to_string();
        }

        if matches!(ext.as_str(), "json" | "xml" | "yaml" | "yml" | "toml" | "ini") {
            return "configuración".to_string();
        }

        "archivo".to_string()
    }

    /// Verifica si un archivo debe ser indexado
    fn should_index(&self, path: &Path) -> bool {
        // Verificar extensión
        let ext = path
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();

        if !self.config.allowed_extensions.is_empty()
            && !self.config.allowed_extensions.contains(&ext)
        {
            return false;
        }

        // Verificar patrones de exclusión
        let path_str = path.to_string_lossy().to_string();
        for pattern in &self.config.exclude_patterns {
            // Matching simple de patrones
            if path_str.contains(pattern.trim_matches('*').trim_matches('/')) {
                return false;
            }
        }

        // Verificar que no sea un archivo oculto (en Unix)
        if let Some(name) = path.file_name() {
            let name = name.to_string_lossy();
            if name.starts_with('.') {
                return false;
            }
        }

        true
    }

    /// Calcula el hash SHA-256 de un archivo de forma streaming (no carga todo en RAM)
    fn compute_hash(&self, path: &Path) -> Result<String> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;
        let mut hasher = Sha256::new();
        let mut buf = [0u8; 65536]; // 64KB buffer
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        Ok(format!("{:x}", hasher.finalize()))
    }
}

/// Resultado de un escaneo de carpeta
#[derive(Debug, Clone)]
pub struct ScanResult {
    pub new_files: u64,
    pub updated_files: u64,
    pub deleted_files: u64,
    pub failed_files: u64,
    pub total_scanned: u64,
}
