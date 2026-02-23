use crate::config::SearchConfig;
use crate::embeddings::OllamaClient;
use crate::error::Result;
use crate::models::*;
use crate::storage::SqliteStorage;
use crate::vector_store::VectorStore;
use tracing::{debug, info};

/// Stop words en español — palabras demasiado comunes que contaminan FTS
const STOP_WORDS_ES: &[&str] = &[
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "aquel", "aquellas",
    "aquellos", "aqui", "asi", "como", "con", "cual", "cuando", "de", "del", "desde",
    "donde", "el", "ella", "ellas", "ellos", "en", "entre", "era", "esa", "esas",
    "ese", "eso", "esos", "esta", "estas", "este", "esto", "estos", "fue", "ha",
    "hay", "la", "las", "le", "les", "lo", "los", "mas", "me", "mi", "muy",
    "no", "nos", "o", "otra", "otras", "otro", "otros", "para", "pero", "por",
    "que", "quien", "se", "si", "sin", "sobre", "somos", "son", "su", "sus",
    "te", "ti", "tu", "tus", "un", "una", "unas", "uno", "unos", "y", "ya", "yo",
];

/// Mapeo de palabras de intención → extensiones de archivo.
/// Permite detección instantánea de tipo de archivo sin llamar al LLM.
///
/// Cada entrada es: (keywords_que_activan, extensiones_resultantes, es_filtro_duro)
/// is_hard_filter=true solo cuando la palabra es inequívoca (ej: "foto", "imagen")
const TYPE_INTENT_MAP: &[(&[&str], &[&str], bool)] = &[
    // Imágenes
    (
        &["imagen", "imagenes", "imágenes", "foto", "fotos", "fotografía",
          "fotografias", "fotografías", "picture", "photo", "img", "screenshot",
          "captura", "pantalla"],
        &["jpg", "jpeg", "png", "gif", "webp", "svg", "bmp", "tiff", "tif", "heic"],
        true,  // filtro duro: si dices "foto" claramente buscas imagen
    ),
    // Documentos Word (intención explícita)
    (
        &["word", "doc", "docx", "odt", "rtf"],
        &["docx", "doc", "odt", "rtf"],
        true,
    ),
    // Documentos (intención genérica)
    (
        &["documento", "documentos", "carta", "oficio",
          "escrito", "texto", "redaccion", "redacción"],
        &["docx", "doc", "odt", "rtf"],
        false, // no duro: "documento" puede querer decir también PDF
    ),
    // PDF
    (
        &["pdf", "escaneado", "escaneo", "acrobat"],
        &["pdf"],
        true,
    ),
    // Excel / hojas de cálculo
    (
        &["excel", "hoja", "xlsx", "calculo", "cálculo", "tabla", "datos",
          "spreadsheet", "csv", "planilla"],
        &["xlsx", "xls", "csv", "tsv", "ods"],
        false,
    ),
    // Presentaciones
    (
        &["presentacion", "presentación", "diapositiva", "diapositivas",
          "powerpoint", "pptx", "slide", "slides"],
        &["pptx", "ppt", "odp"],
        true,
    ),
    // Código
    (
        &["codigo", "código", "script", "programa", "fuente", "source",
          "python", "javascript", "rust", "java"],
        &["py", "js", "ts", "rs", "go", "java", "cpp", "c", "h", "sh",
          "rb", "php", "swift", "kt"],
        false,
    ),
    // Texto plano
    (
        &["txt", "texto plano", "log", "notas", "nota"],
        &["txt", "md", "log"],
        false,
    ),
    // Comprimidos
    (
        &["zip", "comprimido", "archivo", "carpeta comprimida", "rar", "tar"],
        &["zip", "rar", "tar", "gz", "7z"],
        true,
    ),
];

/// Motor de búsqueda que combina búsqueda semántica (vectorial) con FTS
pub struct SearchEngine<'a> {
    storage: &'a SqliteStorage,
    vector_store: &'a dyn VectorStore,
    ollama: &'a OllamaClient,
    config: SearchConfig,
}

impl<'a> SearchEngine<'a> {
    pub fn new(
        storage: &'a SqliteStorage,
        vector_store: &'a dyn VectorStore,
        ollama: &'a OllamaClient,
        config: SearchConfig,
    ) -> Self {
        Self {
            storage,
            vector_store,
            ollama,
            config,
        }
    }

    /// Limpia una consulta para FTS: elimina stop words y caracteres especiales
    fn clean_fts_query(query: &str) -> String {
        let words: Vec<&str> = query
            .split_whitespace()
            .filter(|w| {
                let lower = w.to_lowercase();
                lower.len() > 2 && !STOP_WORDS_ES.contains(&lower.as_str())
            })
            .collect();

        if words.is_empty() {
            query.chars()
                .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                .collect::<String>()
                .trim()
                .to_string()
        } else {
            words.join(" OR ")
        }
    }

    /// Extrae tokens "de memoria" desde consulta en lenguaje natural
    /// (palabras informativas que pueden aparecer en nombre/keywords/descripción).
    fn memory_tokens(query: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for token in query
            .split(|c: char| !c.is_alphanumeric())
            .map(|w| w.trim().to_lowercase())
            .filter(|w| w.len() >= 3)
        {
            if STOP_WORDS_ES.contains(&token.as_str()) {
                continue;
            }
            if seen.insert(token.clone()) {
                out.push(token);
            }
        }

        out
    }

    /// Re-ranking léxico para consultas de recuerdo humano.
    /// Mejora casos como: "mi credencial del ine", "por detrás", etc.
    fn apply_memory_recall_boost(mut results: Vec<SearchResult>, query_text: &str) -> Vec<SearchResult> {
        if results.is_empty() {
            return results;
        }

        let tokens = Self::memory_tokens(query_text);
        if tokens.is_empty() {
            return results;
        }

        let q = query_text.to_lowercase();
        let asks_back = ["detras", "detrás", "posterior", "reverso", "atras", "atrás", "back"]
            .iter()
            .any(|k| q.contains(k));
        let asks_front = ["frente", "frontal", "anverso", "front"]
            .iter()
            .any(|k| q.contains(k));

        for r in &mut results {
            let desc = r.file.metadata.description.as_deref().unwrap_or("");
            let title = r.file.metadata.title.as_deref().unwrap_or("");
            let keywords = r.file.metadata.keywords.join(" ");
            let haystack = format!(
                "{} {} {} {} {}",
                r.file.filename,
                title,
                desc,
                r.file.content_preview,
                keywords
            )
            .to_lowercase();

            let token_hits = tokens.iter().filter(|t| haystack.contains(t.as_str())).count();
            let lexical_bonus = (token_hits as f32 * 0.03).min(0.15);
            r.score = (r.score + lexical_bonus).min(1.0);

            if asks_back {
                let back_hit = ["posterior", "reverso", "detras", "detrás", "atras", "atrás", "back"]
                    .iter()
                    .any(|k| haystack.contains(k));
                if back_hit {
                    r.score = (r.score + 0.08).min(1.0);
                }
            }

            if asks_front {
                let front_hit = ["frente", "frontal", "anverso", "front"]
                    .iter()
                    .any(|k| haystack.contains(k));
                if front_hit {
                    r.score = (r.score + 0.08).min(1.0);
                }
            }
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results
    }

    /// Detecta intención de tipo de archivo usando reglas locales (sin LLM).
    /// Devuelve (extensiones_detectadas, es_filtro_duro).
    /// Rápido y determinista — complementa al LLM.
    fn detect_type_intent(query: &str) -> (Vec<String>, bool) {
        let lower = query.to_lowercase();
        let mut detected_extensions: Vec<String> = Vec::new();
        let mut is_hard = false;

        for (keywords, extensions, hard) in TYPE_INTENT_MAP {
            let matched = keywords.iter().any(|kw| {
                // Buscar la palabra como token (no como substring de otra)
                lower.split_whitespace().any(|word| {
                    let word = word.trim_matches(|c: char| !c.is_alphanumeric());
                    word == *kw
                }) || lower.contains(&format!(" {} ", kw))
                    || lower.starts_with(&format!("{} ", kw))
                    || lower.ends_with(&format!(" {}", kw))
                    || lower == *kw
            });

            if matched {
                for ext in *extensions {
                    if !detected_extensions.contains(&ext.to_string()) {
                        detected_extensions.push(ext.to_string());
                    }
                }
                if *hard {
                    is_hard = true;
                }
            }
        }

        (detected_extensions, is_hard)
    }

    /// Búsqueda principal: combina semántica + FTS + filtros
    ///
    /// Flujo optimizado para latencia mínima:
    /// 1. Detección de tipo por reglas (instantáneo)
    /// 2. EN PARALELO: búsqueda semántica con query original + LLM extrae keywords
    /// 3. FTS con keywords del LLM
    /// 4. Fusionar + filtrar + cortar
    ///
    /// El cuello de botella anterior era esperar al LLM (~16s) ANTES de buscar.
    /// Ahora la búsqueda semántica corre en paralelo con el LLM, así el tiempo
    /// total es max(embedding, llm) en vez de embedding + llm.
    pub async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>> {
        info!("Búsqueda: \"{}\"", query.text);
        let search_start = std::time::Instant::now();

        // ── 1. Detección rápida de tipo por reglas (sin LLM) ──────────────────
        let (rule_extensions, rule_is_hard) = Self::detect_type_intent(&query.text);
        if !rule_extensions.is_empty() {
            debug!(
                "Intención de tipo detectada por reglas: {:?} (filtro_duro={})",
                rule_extensions, rule_is_hard
            );
        }

        // ── 2. EN PARALELO: semántica + LLM enhancement ──────────────────────
        // La búsqueda semántica usa la query ORIGINAL del usuario (nomic-embed-text
        // ya maneja lenguaje natural bien). El LLM solo aporta keywords para FTS
        // y detección de tipo — no necesitamos esperar al LLM para buscar.
        // Pedir más candidatos semánticos que el límite final para mejorar recall.
        // Con límite estricto (ej: 5), archivos relevantes pueden quedar fuera antes
        // de fusionar con FTS (ej: ine_posterior solo aparece por FTS).
        let semantic_future = self.search_semantic(&query.text, query.limit + 20, query.min_score);

        let enhance_future = async {
            if self.config.use_query_enhancement {
                match self.ollama.enhance_search_query(&query.text).await {
                    Ok(eq) => {
                        info!(
                            "🧠 LLM keywords={:?}, tipos={:?}, hard={}",
                            eq.keywords, eq.file_types, eq.hard_type_filter
                        );
                        Some(eq)
                    }
                    Err(e) => {
                        debug!("No se pudo mejorar la consulta: {}", e);
                        None
                    }
                }
            } else {
                None
            }
        };

        // Ejecutar ambas operaciones en paralelo
        let (semantic_result, enhanced) = tokio::join!(semantic_future, enhance_future);
        let semantic_results = semantic_result?;

        let parallel_elapsed = search_start.elapsed();
        debug!("Fase paralela completada en {:.1}s", parallel_elapsed.as_secs_f64());

        // Log de scores semánticos para diagnóstico
        if !semantic_results.is_empty() {
            info!("🔢 Resultados semánticos (top {}):", semantic_results.len().min(10));
            for (i, r) in semantic_results.iter().take(10).enumerate() {
                info!(
                    "   {}. {:.4} {} ({})",
                    i + 1, r.score, r.file.filename, r.file.extension
                );
            }
        } else {
            info!("🔢 Sin resultados semánticos");
        }

        // ── 3. Fusionar extensiones detectadas (reglas + LLM) ─────────────────
        let mut combined_extensions: Vec<String> = rule_extensions.clone();
        let mut is_hard_filter = rule_is_hard;

        if let Some(ref eq) = enhanced {
            for ext in &eq.file_types {
                let ext_lower = ext.to_lowercase();
                if !combined_extensions.contains(&ext_lower) {
                    combined_extensions.push(ext_lower);
                }
            }
            // Seguridad: SOLO reglas locales activan filtro duro.
            // El LLM puede alucinar tipos (ej: pdf/doc/xls/txt) y dejar en 0 resultados.
            // Sus extensiones se usan como señal suave (bonus), nunca exclusión estricta.
            if rule_is_hard {
                is_hard_filter = true;
            }
        }

        // ── 4. Búsqueda full-text ──────────────────────────────────────────
        // Usar keywords del LLM si los hay, si no usar la consulta limpia
        let fts_text = enhanced
            .as_ref()
            .filter(|e| !e.keywords.is_empty())
            .map(|e| e.keywords.join(" OR "))
            .unwrap_or_else(|| Self::clean_fts_query(&query.text));

        let fts_results = if !fts_text.is_empty() {
            self.search_fts(&fts_text, query.limit)?
        } else {
            Vec::new()
        };

        if !fts_results.is_empty() {
            info!("📝 Resultados FTS ({}):", fts_results.len());
            for (i, r) in fts_results.iter().take(5).enumerate() {
                info!("   {}. {:.4} {}", i + 1, r.score, r.file.filename);
            }
        }

        // ── 5. Fusionar resultados ────────────────────────────────────────────
        let mut combined = self.fuse_results(
            semantic_results,
            fts_results,
            query.limit,
            query.min_score,
        );

        // Log post-fusión
        if !combined.is_empty() {
            info!("🔀 Post-fusión (top {}):", combined.len().min(5));
            for (i, r) in combined.iter().take(5).enumerate() {
                info!("   {}. {:.4} {} ({})", i + 1, r.score, r.file.filename, r.file.extension);
            }
        }

        // ── 6. Aplicar filtros de usuario ─────────────────────────────────────
        if !query.filters.extensions.is_empty()
            || !query.filters.mime_types.is_empty()
            || !query.filters.category_ids.is_empty()
            || query.filters.date_from.is_some()
            || query.filters.date_to.is_some()
            || query.filters.min_size.is_some()
            || query.filters.max_size.is_some()
            || query.filters.path_contains.is_some()
        {
            combined = self.apply_filters(combined, &query.filters);
        }

        // ── 7. Filtro de tipo de archivo ──────────────────────────────────────
        if !combined_extensions.is_empty() {
            if is_hard_filter {
                // Filtro duro: solo devolver archivos del tipo detectado
                combined.retain(|r| {
                    combined_extensions
                        .iter()
                        .any(|ext| r.file.extension.eq_ignore_ascii_case(ext))
                });
                debug!(
                    "Filtro duro aplicado ({:?}): {} resultados quedan",
                    combined_extensions,
                    combined.len()
                );
            } else {
                // Filtro suave: bonus ADITIVO si coincide con el tipo detectado.
                // 1.3x multiplicativo creaba gaps artificiales que el gap cutter amplificaba,
                // eliminando resultados relevantes. Bonus aditivo de 0.04 es más estable.
                combined = combined
                    .into_iter()
                    .map(|mut r| {
                        if combined_extensions
                            .iter()
                            .any(|ext| r.file.extension.eq_ignore_ascii_case(ext))
                        {
                            r.score = (r.score + 0.04).min(1.0);
                        }
                        r
                    })
                    .collect();
                combined.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            }
        }

        // ── 7.5 Re-ranking por señales de "recuerdo" en lenguaje natural ──
        combined = Self::apply_memory_recall_boost(combined, &query.text);

        // ── 8. Cortar por brecha de relevancia ────────────────────────────────
        combined = Self::cut_at_relevance_gap(combined);

        // ── 9. Agregar información virtual ───────────────────────────────────
        for result in &mut combined {
            if let Ok(vfiles) = self.storage.get_virtual_files_for_file(&result.file.id) {
                result.virtual_info = vfiles.into_iter().next();
            }
        }

        info!("Búsqueda completada: {} resultados en {:.1}s", combined.len(), search_start.elapsed().as_secs_f64());
        Ok(combined)
    }

    /// Detecta la brecha más grande de relevancia entre resultados consecutivos
    /// y corta ahí. Esto elimina el "ruido" de archivos con score bajo que no son
    /// realmente relevantes a la consulta.
    ///
    /// Estrategia: score floor primero (eliminar basura), luego gap detection.
    /// Garantiza mínimo 2 resultados si están por encima del floor.
    fn cut_at_relevance_gap(mut results: Vec<SearchResult>) -> Vec<SearchResult> {
        if results.len() <= 1 {
            return results;
        }

        let top_score = results[0].score;
        if top_score <= 0.0 {
            return results;
        }

        // ── 1. Score floor (primero) ─────────────────────────────────────
        // Eliminar resultados claramente irrelevantes antes de buscar gaps.
        // Esto evita que un resultado lejano (0.52) cree un gap artificial
        // que enmascare la distribución de los resultados relevantes.
        let score_floor = (top_score * 0.82).max(0.42);
        let before_floor = results.len();
        results.retain(|r| r.score >= score_floor);
        if results.len() < before_floor {
            debug!(
                "Score floor {:.3} eliminó {} resultados irrelevantes",
                score_floor,
                before_floor - results.len()
            );
        }

        if results.len() <= 2 {
            return results;
        }

        // ── 2. Gap detection (después del floor) ─────────────────────────
        // Buscar la mayor brecha entre resultados consecutivos.
        // Umbral: 12% relativo al score en esa posición (mín absoluto 0.06).
        // 3% era demasiado bajo: un gap de 0.08 en scores de 0.79→0.71
        // cortaba informes de actividades que TODOS eran relevantes.
        let mut max_gap = 0.0f32;
        let mut cut_index = results.len();

        for i in 0..results.len() - 1 {
            let gap = results[i].score - results[i + 1].score;
            let relative_threshold = (results[i].score * 0.12).max(0.06);
            if gap > max_gap && gap >= relative_threshold {
                max_gap = gap;
                cut_index = i + 1;
            }
        }

        // Garantizar mínimo 2 resultados: si el gap cortaría a solo 1,
        // solo cortar si la brecha es realmente enorme (>20%).
        if cut_index == 1 {
            let gap_ratio = max_gap / results[0].score;
            if gap_ratio < 0.20 {
                cut_index = results.len(); // no cortar
            }
        }

        if cut_index < results.len() && cut_index >= 2 {
            debug!(
                "Cortando en posición {} (gap={:.4}, scores: {:.4} → {:.4})",
                cut_index,
                max_gap,
                results[cut_index - 1].score,
                results[cut_index].score,
            );
            results.truncate(cut_index);
        }

        results
    }

    /// Construye el snippet más útil para mostrar en resultados de búsqueda.
    ///
    /// Prioridad:
    /// 1. metadata.description (resumen semántico del LLM o visión) — más informativo
    /// 2. content_preview — extracto real del contenido
    /// 3. Nombre del archivo humanizado — último recurso
    ///
    /// Para imágenes, el content_preview suele ser "Resumen visual (fallback)..."
    /// que no es útil. La description (si existe) siempre es mejor.
    fn build_result_snippet(file: &IndexedFile) -> String {
        let is_image = matches!(
            file.extension.as_str(),
            "jpg" | "jpeg" | "png" | "bmp" | "tiff" | "webp" | "gif" | "svg" | "heic"
        );

        // Para imágenes: preferir description ya que content_preview puede ser un
        // fallback genérico del nombre de archivo
        if is_image {
            if let Some(ref desc) = file.metadata.description {
                let d = desc.trim();
                // Preferir description si parece un resumen real (no solo nombre de archivo)
                // Las descripciones buenas de los modelos VL suelen tener >50 chars y frases completas
                if !d.is_empty() && d.len() > 50 {
                    return d.chars().take(200).collect();
                }
            }
            // Intentar content_preview si no es un fallback genérico
            if !file.content_preview.is_empty() {
                let cp = file.content_preview.trim();
                // content_preview con resumen visual real suele ser descriptivo
                if cp.len() > 50 && !cp.starts_with("Imagen:") {
                    return cp.chars().take(200).collect::<String>();
                }
            }
            // Para imágenes sin buena description, construir algo útil
            let mut parts = Vec::new();
            if let Some(ref title) = file.metadata.title {
                if !title.is_empty() && *title != file.filename {
                    parts.push(title.clone());
                }
            }
            if !file.metadata.keywords.is_empty() {
                parts.push(format!("[{}]", file.metadata.keywords.join(", ")));
            }
            if parts.is_empty() {
                // Humanizar el nombre del archivo
                let clean = file.filename
                    .replace(['_', '-', '.'], " ")
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ");
                return clean;
            }
            return parts.join(" — ").chars().take(200).collect();
        }

        // Para documentos: preferir description si es informativa, sino content_preview
        if let Some(ref desc) = file.metadata.description {
            let d = desc.trim();
            if !d.is_empty() && d.len() > 20 {
                return d.chars().take(200).collect();
            }
        }

        if !file.content_preview.is_empty() {
            return file.content_preview
                .chars()
                .take(200)
                .collect::<String>();
        }

        // Último recurso
        file.filename
            .replace(['_', '-', '.'], " ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Búsqueda semántica usando embeddings
    async fn search_semantic(
        &self,
        query: &str,
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<SearchResult>> {
        // Generar embedding de la consulta
        let query_embedding = self.ollama.generate_embedding(query).await?;

        // Buscar en el vector store — pedimos limit+5 para tener margen
        let matches = self
            .vector_store
            .search(&query_embedding, limit + 5, min_score)?;

        // Resolver archivos
        let mut results = Vec::new();
        for m in matches {
            if let Ok(Some(file)) = self.storage.get_file_by_id(&m.file_id) {
                let snippet = Self::build_result_snippet(&file);

                results.push(SearchResult {
                    file,
                    score: m.score,
                    virtual_info: None,
                    snippet,
                    explanation: None,
                });
            }
        }

        Ok(results)
    }

    /// Búsqueda full-text usando SQLite FTS5
    fn search_fts(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let fts_results = self.storage.search_fts(query, limit + 5)?;

        if fts_results.is_empty() {
            return Ok(Vec::new());
        }

        // Encontrar el rango de ranks para normalización relativa
        let max_rank = fts_results
            .iter()
            .map(|(_, r)| r.abs())
            .fold(0.0f64, f64::max);
        let min_rank = fts_results
            .iter()
            .map(|(_, r)| r.abs())
            .fold(f64::MAX, f64::min);

        let results: Vec<SearchResult> = fts_results
            .into_iter()
            .map(|(file, rank)| {
                let snippet = Self::build_result_snippet(&file);

                // Normalizar rank de FTS5 a un score 0-1
                // FTS5 rank: más negativo = más relevante (rank.abs() mayor = más relevante)
                let abs_rank = rank.abs();
                let score = if max_rank > min_rank {
                    // Normalización min-max conservadora
                    let normalized = (abs_rank - min_rank) / (max_rank - min_rank);
                    // Rango [0.05, 0.45]: FTS es señal de apoyo, no principal
                    // Evita que FTS infle scores de resultados semánticamente débiles
                    (normalized * 0.40 + 0.05) as f32
                } else {
                    // Solo un resultado o todos con mismo rank
                    0.25f32
                };

                SearchResult {
                    file,
                    score,
                    virtual_info: None,
                    snippet,
                    explanation: None,
                }
            })
            .collect();

        Ok(results)
    }

    /// Fusiona resultados de búsqueda semántica y FTS.
    ///
    /// REGLA CARDINAL: tener match FTS NUNCA debe empeorar el score.
    /// La fórmula anterior (sem*0.75 + fts*0.25 + 0.08) penalizaba archivos
    /// con match FTS débil. Ejemplo: sem=0.54, fts=0.05 → fused=0.497 < 0.54.
    /// Esto hacía que ine_frente.jpeg (match "ine" en filename) cayera por debajo
    /// de imágenes irrelevantes sin match FTS.
    ///
    /// Estrategia corregida:
    /// - Ambas señales: tomar semántico + bonus aditivo de FTS (solo suma, nunca resta)
    /// - Solo semántico: usar score directo
    /// - Solo FTS: penalizar (sin confirmación semántica = poco fiable)
    fn fuse_results(
        &self,
        semantic: Vec<SearchResult>,
        fts: Vec<SearchResult>,
        limit: usize,
        min_score: f32,
    ) -> Vec<SearchResult> {
        use std::collections::HashMap;

        let mut scores: HashMap<String, (f32, f32, Option<SearchResult>)> = HashMap::new();

        for result in semantic {
            let entry = scores
                .entry(result.file.id.clone())
                .or_insert((0.0, 0.0, None));
            entry.0 = result.score;
            if entry.2.is_none() {
                entry.2 = Some(result);
            }
        }

        for result in fts {
            let entry = scores
                .entry(result.file.id.clone())
                .or_insert((0.0, 0.0, None));
            entry.1 = result.score;
            if entry.2.is_none() {
                entry.2 = Some(result);
            }
        }

        let mut combined: Vec<SearchResult> = scores
            .into_values()
            .filter_map(|(sem_score, fts_score, result)| {
                let final_score = if sem_score > 0.0 && fts_score > 0.0 {
                    // Ambas señales: semántico como base + bonus aditivo de FTS.
                    // FTS bonus: entre 0.02 (match débil) y 0.08 (match fuerte).
                    // NUNCA puede bajar el score semántico.
                    let fts_bonus = fts_score * 0.15 + 0.02;
                    (sem_score + fts_bonus).min(1.0)
                } else if sem_score > 0.0 {
                    // Solo semántico: score directo
                    sem_score
                } else {
                    // Solo FTS: señal léxica fuerte (nombres/ruta/contenido exacto).
                    // Antes (*0.40) era demasiado bajo y expulsaba resultados válidos
                    // como "ine_posterior.jpeg" aunque fueran match claro.
                    // Mantenerlo por debajo de semántico alto, pero competitivo.
                    (fts_score * 0.90 + 0.12).min(0.65)
                };

                if final_score < min_score {
                    return None;
                }

                result.map(|mut r| {
                    r.score = final_score;
                    r
                })
            })
            .collect();

        combined.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        combined.truncate(limit);
        combined
    }

    /// Aplica filtros a los resultados
    fn apply_filters(
        &self,
        results: Vec<SearchResult>,
        filters: &SearchFilters,
    ) -> Vec<SearchResult> {
        results
            .into_iter()
            .filter(|r| {
                // Filtrar por extensión
                if !filters.extensions.is_empty()
                    && !filters
                        .extensions
                        .iter()
                        .any(|ext| r.file.extension.eq_ignore_ascii_case(ext))
                {
                    return false;
                }

                // Filtrar por MIME type
                if !filters.mime_types.is_empty()
                    && !filters.mime_types.contains(&r.file.mime_type)
                {
                    return false;
                }

                // Filtrar por fecha
                if let Some(ref from) = filters.date_from {
                    if r.file.modified_at < *from {
                        return false;
                    }
                }
                if let Some(ref to) = filters.date_to {
                    if r.file.modified_at > *to {
                        return false;
                    }
                }

                // Filtrar por tamaño
                if let Some(min) = filters.min_size {
                    if r.file.size < min {
                        return false;
                    }
                }
                if let Some(max) = filters.max_size {
                    if r.file.size > max {
                        return false;
                    }
                }

                // Filtrar por ruta
                if let Some(ref path_contains) = filters.path_contains {
                    if !r
                        .file
                        .path
                        .to_string_lossy()
                        .to_lowercase()
                        .contains(&path_contains.to_lowercase())
                    {
                        return false;
                    }
                }

                true
            })
            .collect()
    }
}
