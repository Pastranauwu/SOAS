use soas_core::prelude::*;
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::history::DefaultHistory;
use rustyline::validate::Validator;
use rustyline::{Context, Editor, Helper};
use soas_core::indexer::pipeline::ScanResult;
use std::io::{self, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CliMode {
    Normal,
    Debug,
}

impl CliMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Debug => "debug",
        }
    }

    fn from_value(value: &str) -> Option<Self> {
        match value.trim().to_lowercase().as_str() {
            "normal" => Some(Self::Normal),
            "debug" => Some(Self::Debug),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CliOptions {
    mode: CliMode,
    show_help: bool,
}

const BASE_COMMANDS: &[&str] = &[
    "help",
    "stats",
    "cats",
    "categorias",
    "rescan",
    "rebuild",
    "reindex",
    "reimages",
    "carpetas",
    "agregar ",
    "add ",
    "quitar ",
    "remove ",
    "archivos",
    "files",
    "ls",
    "info ",
    "borrar ",
    "delete ",
    "salir",
    "exit",
    "q",
];

fn parse_cli_options() -> CliOptions {
    let mut mode = std::env::var("SOAS_CLI_MODE")
        .ok()
        .and_then(|v| CliMode::from_value(&v))
        .unwrap_or(CliMode::Normal);
    let mut show_help = false;

    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--debug" | "-d" => mode = CliMode::Debug,
            "--normal" | "-n" => mode = CliMode::Normal,
            "--help" | "-h" => show_help = true,
            _ => {
                if let Some(value) = arg.strip_prefix("--mode=") {
                    if let Some(parsed) = CliMode::from_value(value) {
                        mode = parsed;
                    }
                }
            }
        }
    }

    CliOptions { mode, show_help }
}

fn print_startup_help() {
    println!("SOAS CLI");
    println!();
    println!("Uso:");
    println!("  soas-cli [--normal|--debug] [--help]");
    println!();
    println!("Modos:");
    println!("  --normal, -n   Modo usuario (por defecto): salida limpia con barra de progreso");
    println!("  --debug,  -d   Modo desarrollador: logs técnicos detallados");
    println!();
    println!("Variables de entorno:");
    println!("  SOAS_CLI_MODE=normal|debug");
    println!();
    println!("Dentro del modo interactivo usa 'help' para ver comandos disponibles.");
}

fn init_logging(mode: CliMode) -> anyhow::Result<()> {
    let base_filter = tracing_subscriber::EnvFilter::from_default_env();
    let directive = match mode {
        CliMode::Normal => "soas_core=error",
        CliMode::Debug => "soas_core=info",
    };

    tracing_subscriber::fmt()
        .with_env_filter(base_filter.add_directive(directive.parse()?))
        .with_target(matches!(mode, CliMode::Debug))
        .without_time()
        .init();

    Ok(())
}

#[derive(Clone, Default)]
struct CliHelper {
    commands: Vec<String>,
}

impl Helper for CliHelper {}
impl Validator for CliHelper {}
impl Highlighter for CliHelper {}

impl Hinter for CliHelper {
    type Hint = String;

    fn hint(&self, _line: &str, _pos: usize, _ctx: &Context<'_>) -> Option<String> {
        None
    }
}

impl Completer for CliHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let prefix = &line[..pos];
        let mut candidates: Vec<Pair> = self
            .commands
            .iter()
            .filter(|cmd| cmd.starts_with(prefix))
            .take(30)
            .map(|cmd| Pair {
                display: cmd.clone(),
                replacement: cmd.clone(),
            })
            .collect();

        if candidates.is_empty() {
            candidates = self
                .commands
                .iter()
                .filter(|cmd| cmd.contains(prefix))
                .take(30)
                .map(|cmd| Pair {
                    display: cmd.clone(),
                    replacement: cmd.clone(),
                })
                .collect();
        }

        Ok((0, candidates))
    }
}

fn detect_default_scan_path() -> Option<String> {
    // Prioridad: carpeta de Descargas nativa → Documentos → Home
    // `dirs` resuelve automáticamente la ruta correcta según el SO:
    //   Linux  → $HOME/Descargas  o  $HOME/Downloads
    //   Windows → C:\Users\<user>\Downloads
    //   macOS  → /Users/<user>/Downloads
    if let Some(downloads) = dirs::download_dir() {
        if downloads.exists() && downloads.is_dir() {
            return Some(downloads.to_string_lossy().to_string());
        }
    }

    if let Some(documents) = dirs::document_dir() {
        if documents.exists() && documents.is_dir() {
            return Some(documents.to_string_lossy().to_string());
        }
    }

    // Último recurso: home del usuario
    if let Some(home) = dirs::home_dir() {
        if home.exists() && home.is_dir() {
            return Some(home.to_string_lossy().to_string());
        }
    }

    None
}

fn read_stdin_line() -> anyhow::Result<String> {
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

async fn ensure_first_folder_if_empty(soas: &mut Soas) -> anyhow::Result<()> {
    let stats = soas.stats()?;
    let folders = soas.list_folders()?;

    if stats.total_files > 0 || !folders.is_empty() {
        return Ok(());
    }

    let default_path = detect_default_scan_path().unwrap_or_else(|| "./".to_string());

    println!("📂 Base de datos vacía: elige una carpeta inicial para indexar.");
    println!("   Presiona Enter para usar: {}", default_path);
    println!("   Escribe 'salir' para omitir por ahora.");

    loop {
        print!("📁 Carpeta a escanear: ");
        io::stdout().flush()?;
        let path_input = read_stdin_line()?;

        if path_input.eq_ignore_ascii_case("salir") || path_input.eq_ignore_ascii_case("exit") {
            println!("   ⚠️  Inicio sin carpeta. Usa 'agregar <ruta>' después.");
            println!();
            return Ok(());
        }

        let selected_path = if path_input.is_empty() {
            default_path.clone()
        } else {
            path_input
        };

        let candidate = Path::new(&selected_path);
        if !candidate.exists() || !candidate.is_dir() {
            println!("   ❌ Ruta inválida. Ingresa una carpeta existente.");
            continue;
        }

        let name = candidate
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .filter(|n| !n.trim().is_empty())
            .unwrap_or_else(|| "Carpeta inicial".to_string());

        soas.add_folder(&selected_path, &name).await?;
        println!("   ✅ Carpeta '{}' agregada: {}", name, selected_path);
        println!();
        return Ok(());
    }
}

fn print_scan_summary(results: &[ScanResult], elapsed_secs: f64) {
    for result in results {
        println!("   📊 Resultado del escaneo:");
        println!("      Nuevos:       {}", result.new_files);
        println!("      Actualizados: {}", result.updated_files);
        println!("      Eliminados:   {}", result.deleted_files);
        println!("      Errores:      {}", result.failed_files);
        println!("      Total:        {}", result.total_scanned);
    }
    println!("   ⏱️  Tiempo: {:.1}s", elapsed_secs);
    println!();
}

fn render_progress(progress: &IndexProgress, last_line: &Arc<Mutex<String>>, mode: CliMode) {
    if progress.total == 0 {
        return;
    }

    let processed = (progress.processed + 1).min(progress.total);
    let percent = (processed as f64 / progress.total as f64 * 100.0).round() as u64;
    let bar_width = 28usize;
    let filled = ((processed as f64 / progress.total as f64) * bar_width as f64).round() as usize;
    let mut bar = String::with_capacity(bar_width);
    for _ in 0..filled.min(bar_width) {
        bar.push('█');
    }
    for _ in filled.min(bar_width)..bar_width {
        bar.push('░');
    }

    let line = if matches!(mode, CliMode::Debug) {
        let file_name = progress
            .current_file
            .as_ref()
            .and_then(|p| Path::new(p).file_name().map(|f| f.to_string_lossy().to_string()))
            .unwrap_or_else(|| "archivo".to_string());
        let short_name: String = file_name.chars().take(42).collect();
        format!(
            "   [{}] {:>3}% ({}/{}) | {} | errores: {}",
            bar, percent, processed, progress.total, short_name, progress.failed
        )
    } else {
        format!(
            "   Analizando archivos [{}] {:>3}% ({}/{}) | errores: {}",
            bar, percent, processed, progress.total, progress.failed
        )
    };

    if let Ok(mut guard) = last_line.lock() {
        *guard = line.clone();
    }

    print!("\r{}\x1b[K", line);
    let _ = io::stdout().flush();
}

async fn run_scan_with_progress(soas: &mut Soas, mode: CliMode) -> anyhow::Result<Vec<ScanResult>> {
    let last_line = Arc::new(Mutex::new(String::new()));
    let progress_state = Arc::clone(&last_line);
    let callback = move |progress: IndexProgress| {
        render_progress(&progress, &progress_state, mode);
    };

    let results = soas.scan_all_with_progress(Some(&callback)).await?;
    if let Ok(guard) = last_line.lock() {
        if !guard.is_empty() {
            println!();
        }
    }

    Ok(results)
}

fn build_completion_candidates(soas: &Soas) -> Vec<String> {
    let mut commands: Vec<String> = BASE_COMMANDS.iter().map(|c| c.to_string()).collect();

    if let Ok(files) = soas.list_files() {
        for idx in 1..=files.len() {
            commands.push(format!("info {}", idx));
            commands.push(format!("borrar {}", idx));
        }
    }

    if let Ok(folders) = soas.list_folders() {
        for idx in 1..=folders.len() {
            commands.push(format!("quitar {}", idx));
        }
    }

    commands.sort();
    commands.dedup();
    commands
}

fn print_help() {
    println!("═══════════════════════════════════════════");
    println!("  Búsqueda interactiva (CLI)");
    println!("  Busca en lenguaje natural. Ejemplos:");
    println!("   • \"foto de perfil con fondo blanco\"");
    println!("   • \"contrato en pdf de 2024\"");
    println!("   • \"presentación sobre presupuesto\"");
    println!("  Comandos especiales:");
    println!("   help | stats | cats | rescan | rebuild | reindex | reimages | salir");
    println!("   carpetas         = lista carpetas monitoreadas");
    println!("   agregar <ruta>   = agrega una carpeta para indexar");
    println!("   quitar <num>     = elimina carpeta y sus archivos del índice");
    println!("   archivos         = lista todos los archivos indexados");
    println!("   info <num>       = muestra detalles de un archivo");
    println!("   borrar <num>     = elimina un archivo del índice");
    println!("   🔄 rebuild       = regenera embeddings");
    println!("   🖼️  reimages     = re-analiza solo imágenes");
    println!("   TAB              = autocompleta comandos dinámicos");
    println!("   Modo CLI         = --normal (limpio) | --debug (logs técnicos)");
    println!("═══════════════════════════════════════════");
    println!();
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let options = parse_cli_options();
    if options.show_help {
        print_startup_help();
        return Ok(());
    }

    init_logging(options.mode)?;

    println!("╔══════════════════════════════════════════╗");
    println!("║                SOAS CLI                  ║");
    println!("║   Sistema Inteligente de Búsqueda        ║");
    println!("╚══════════════════════════════════════════╝");
    println!("Modo: {}", options.mode.as_str());
    println!("Tip: escribe 'help' para ver comandos.");
    println!();

    println!("⏳ Inicializando SOAS...");
    let mut soas = Soas::new(SoasConfig::default())?;
    println!("✅ SOAS inicializado");

    // 2. Verificar Ollama
    print!("⏳ Verificando conexión con Ollama... ");
    io::stdout().flush()?;
    match soas.check_ollama().await {
        Ok(true) => println!("✅ Conectado"),
        Ok(false) => {
            println!("❌ No disponible");
            println!("   Asegúrate de que Ollama esté corriendo: ollama serve");
            return Ok(());
        }
        Err(e) => {
            println!("❌ Error: {}", e);
            return Ok(());
        }
    }

    // 3. Listar modelos
    print!("⏳ Verificando modelos... ");
    io::stdout().flush()?;
    let models = soas.list_ollama_models().await?;
    println!("✅ {} modelos disponibles", models.len());

    let required = ["nomic-embed-text", "qwen3:1.7b", "llava-phi3"];
    for model in &required {
        let found = models.iter().any(|m| m.contains(model));
        if found {
            println!("   ✅ {}", model);
        } else {
            println!("   ⚠️  {} no encontrado (ollama pull {})", model, model);
        }
    }
    println!();

    ensure_first_folder_if_empty(&mut soas).await?;

    let folders = soas.list_folders()?;
    if folders.is_empty() {
        println!("⚠️  No hay carpetas registradas todavía.");
        println!("   Usa 'agregar <ruta>' en el modo interactivo para comenzar.");
        println!();
    } else {
        println!("📁 Carpetas registradas: {}", folders.len());
        for f in folders.iter().take(3) {
            println!("   • {}", f.path.display());
        }
        if folders.len() > 3 {
            println!("   • ... y {} más", folders.len() - 3);
        }
        println!();
    }

    let vfs = soas.virtual_fs();
    let existing_cats = soas.stats()?.total_categories;
    if existing_cats == 0 {
        print!("⏳ Creando categorías por defecto... ");
        io::stdout().flush()?;
        let cats = vfs.create_default_categories()?;
        println!("✅ {} categorías creadas", cats.len());
    } else {
        println!("📂 {} categorías existentes", existing_cats);
    }
    println!();

    if !folders.is_empty() {
        println!("🔍 Escaneando e indexando carpetas registradas...");
        println!("   Solo se procesarán archivos nuevos o modificados.");
        println!();

        let start = Instant::now();
        let results = run_scan_with_progress(&mut soas, options.mode).await?;
        let elapsed = start.elapsed();
        print_scan_summary(&results, elapsed.as_secs_f64());
    }

    let stats = soas.stats()?;
    println!("📈 Estadísticas del sistema:");
    println!("   Archivos indexados:  {}", stats.total_files);
    println!("   Con embeddings:      {}", stats.total_embedded);
    println!("   Categorías:          {}", stats.total_categories);
    println!("   Carpetas vigiladas:  {}", stats.total_watched_folders);
    println!(
        "   Tamaño archivos:    {:.1} MB",
        stats.indexed_files_size_bytes as f64 / 1_048_576.0
    );
    println!();

    print_help();

    let mut rl: Editor<CliHelper, DefaultHistory> = Editor::new()?;

    loop {
        let helper = CliHelper {
            commands: build_completion_candidates(&soas),
        };
        rl.set_helper(Some(helper));

        let query = match rl.readline("🔎 Buscar: ") {
            Ok(line) => {
                let trimmed = line.trim().to_string();
                if !trimmed.is_empty() {
                    let _ = rl.add_history_entry(trimmed.as_str());
                }
                trimmed
            }
            Err(ReadlineError::Interrupted) => {
                println!("\n👋 ¡Hasta luego!");
                break;
            }
            Err(ReadlineError::Eof) => {
                println!();
                println!("👋 ¡Hasta luego!");
                break;
            }
            Err(e) => {
                println!("   ❌ Error leyendo entrada: {}", e);
                continue;
            }
        };

        if query.is_empty() {
            continue;
        }
        if query == "salir" || query == "exit" || query == "q" {
            break;
        }

        if query == "help" || query == "ayuda" {
            print_help();
            continue;
        }

        if query == "stats" {
            let s = soas.stats()?;
            println!("   Archivos: {} | Embeddings: {} | Categorías: {}",
                s.total_files, s.total_embedded, s.total_categories);
            println!();
            continue;
        }

        if query == "cats" || query == "categorias" {
            let vfs = soas.virtual_fs();
            let tree = vfs.get_category_tree()?;
            for node in &tree {
                println!("   {} {} ({} archivos)",
                    node.category.icon.as_deref().unwrap_or("📁"),
                    node.category.name,
                    node.file_count,
                );
            }
            println!();
            continue;
        }

        if query.starts_with("rescan") {
            println!("   ⏳ Re-escaneando...");
            let start = Instant::now();
            let results = run_scan_with_progress(&mut soas, options.mode).await?;
            let elapsed = start.elapsed();
            print_scan_summary(&results, elapsed.as_secs_f64());
            continue;
        }

        if query == "rebuild" {
            println!("   🔄 Regenerando embeddings (sin re-extraer contenido)...");
            println!("   Esto regenera los vectores de búsqueda para mejorar resultados.");
            let start = Instant::now();
            match soas.rebuild_embeddings().await {
                Ok(count) => {
                    let elapsed = start.elapsed();
                    println!("   ✅ {} embeddings regenerados en {:.1}s", count, elapsed.as_secs_f64());
                }
                Err(e) => println!("   ❌ Error: {}", e),
            }
            println!();
            continue;
        }

        if query == "reindex" {
            println!("   ♻️  Re-indexación completa (re-extrae contenido + genera embeddings)...");
            println!("   ⚠️  Esto puede tomar varios minutos.");
            let start = Instant::now();
            match soas.reindex_all().await {
                Ok(results) => {
                    let elapsed = start.elapsed();
                    for r in &results {
                        println!("   Procesados: {} | Errores: {} | ⏱️ {:.1}s",
                            r.total_scanned, r.failed_files, elapsed.as_secs_f64());
                    }
                }
                Err(e) => println!("   ❌ Error: {}", e),
            }
            println!();
            continue;
        }

        if query == "reimages" {
            println!("   🖼️  Re-analizando solo imágenes (visión + metadatos LLM)...");
            println!("   Esto es más rápido que reindex completo.");
            let start = Instant::now();
            match soas.reindex_images().await {
                Ok(results) => {
                    let elapsed = start.elapsed();
                    for r in &results {
                        println!("   Actualizados: {} | Errores: {} | ⏱️ {:.1}s",
                            r.updated_files, r.failed_files, elapsed.as_secs_f64());
                    }
                }
                Err(e) => println!("   ❌ Error: {}", e),
            }
            println!();
            continue;
        }

        // ── Gestión de carpetas ──

        if query == "carpetas" || query == "folders" {
            let folders = soas.list_folders()?;
            if folders.is_empty() {
                println!("   No hay carpetas monitoreadas.");
            } else {
                println!("   📁 Carpetas monitoreadas:");
                for (i, f) in folders.iter().enumerate() {
                    let status = if f.active { "✅" } else { "⏸️" };
                    let last = f.last_scan
                        .map(|d| d.format("%Y-%m-%d %H:%M").to_string())
                        .unwrap_or_else(|| "nunca".to_string());
                    println!("   {}. {} {} — {} (último: {})",
                        i + 1, status, f.name, f.path.display(), last);
                }
            }
            println!();
            continue;
        }

        if query.starts_with("agregar ") || query.starts_with("add ") {
            let path = query.splitn(2, ' ').nth(1).unwrap_or("").trim();
            if path.is_empty() {
                println!("   Uso: agregar /ruta/a/carpeta");
            } else {
                let name = std::path::Path::new(path)
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| path.to_string());
                match soas.add_folder(path, &name).await {
                    Ok(f) => {
                        println!("   ✅ Carpeta '{}' agregada", f.name);
                        println!("   Usa 'rescan' para indexar los archivos.");
                    }
                    Err(e) => println!("   ❌ Error: {}", e),
                }
            }
            println!();
            continue;
        }

        if query.starts_with("quitar ") || query.starts_with("remove ") {
            let num_str = query.splitn(2, ' ').nth(1).unwrap_or("").trim();
            let folders = soas.list_folders()?;
            if let Ok(num) = num_str.parse::<usize>() {
                if num >= 1 && num <= folders.len() {
                    let folder = &folders[num - 1];
                    println!("   ⏳ Eliminando '{}' y sus archivos del índice...", folder.name);
                    match soas.remove_folder(&folder.id) {
                        Ok(count) => {
                            println!("   ✅ Carpeta eliminada. {} archivos removidos del índice.", count);
                        }
                        Err(e) => println!("   ❌ Error: {}", e),
                    }
                } else {
                    println!("   Número inválido. Usa 'carpetas' para ver la lista.");
                }
            } else {
                println!("   Uso: quitar <número> — usa 'carpetas' para ver la lista");
            }
            println!();
            continue;
        }

        // ── Gestión de archivos ──

        if query == "archivos" || query == "files" || query == "ls" {
            let files = soas.list_files()?;
            if files.is_empty() {
                println!("   No hay archivos indexados.");
            } else {
                println!("   📄 {} archivos indexados:", files.len());
                for (i, f) in files.iter().enumerate() {
                    let icon = match f.extension.as_str() {
                        "pdf" => "📕",
                        "docx" | "doc" => "📘",
                        "xlsx" | "csv" => "📊",
                        "png" | "jpg" | "jpeg" | "svg" | "webp" => "🖼️",
                        _ => "📄",
                    };
                    let size_kb = f.size as f64 / 1024.0;
                    let status = match &f.index_status {
                        IndexStatus::Indexed => "✅",
                        IndexStatus::Failed(_) => "❌",
                        IndexStatus::Pending => "⏳",
                        IndexStatus::ContentExtracted => "📝",
                        _ => "❓",
                    };
                    println!("   {:>3}. {} {} {} ({:.0} KB) {}",
                        i + 1, status, icon, f.filename, size_kb,
                        f.index_status.as_str());
                }
            }
            println!();
            continue;
        }

        if query.starts_with("info ") {
            let num_str = query.splitn(2, ' ').nth(1).unwrap_or("").trim();
            let files = soas.list_files()?;
            if let Ok(num) = num_str.parse::<usize>() {
                if num >= 1 && num <= files.len() {
                    let f = &files[num - 1];
                    println!("   ╔══ {} ══╗", f.filename);
                    println!("   ║ ID:        {}", f.id);
                    println!("   ║ Ruta:      {}", f.path.display());
                    println!("   ║ Tipo:      {} ({})", f.extension, f.mime_type);
                    println!("   ║ Tamaño:    {:.1} KB", f.size as f64 / 1024.0);
                    println!("   ║ Estado:    {}", f.index_status.as_str());
                    println!("   ║ Creado:    {}", f.created_at.format("%Y-%m-%d %H:%M"));
                    println!("   ║ Modificado: {}", f.modified_at.format("%Y-%m-%d %H:%M"));
                    println!("   ║ Indexado:  {}", f.indexed_at.format("%Y-%m-%d %H:%M"));
                    if let Some(ref title) = f.metadata.title {
                        println!("   ║ Título:    {}", title);
                    }
                    if let Some(ref desc) = f.metadata.description {
                        println!("   ║ Descripción: {}", desc);
                    }
                    if let Some(ref summary) = f.metadata.summary {
                        println!("   ║ Resumen:   {}", summary);
                    }
                    if !f.metadata.keywords.is_empty() {
                        println!("   ║ Keywords:  {}", f.metadata.keywords.join(", "));
                    }
                    if !f.metadata.semantic_tags.is_empty() {
                        println!("   ║ Tags:      {}", f.metadata.semantic_tags.join(", "));
                    }
                    if let Some(ref group) = f.metadata.content_type_group {
                        println!("   ║ Grupo:     {}", group);
                    }
                    if let Some(ref method) = f.metadata.extra.get("ocr") {
                        println!("   ║ OCR:       {}", method);
                    }
                    if !f.content_preview.is_empty() {
                        let preview: String = f.content_preview.chars().take(200).collect();
                        let preview = preview.replace('\n', " ");
                        println!("   ║ Preview:   {}...", preview.trim());
                    }
                    println!("   ╚══════════════════════════╝");
                } else {
                    println!("   Número inválido. Usa 'archivos' para ver la lista.");
                }
            } else {
                println!("   Uso: info <número>");
            }
            println!();
            continue;
        }

        if query.starts_with("borrar ") || query.starts_with("delete ") {
            let num_str = query.splitn(2, ' ').nth(1).unwrap_or("").trim();
            let files = soas.list_files()?;
            if let Ok(num) = num_str.parse::<usize>() {
                if num >= 1 && num <= files.len() {
                    let f = &files[num - 1];
                    println!("   ⏳ Eliminando '{}' del índice...", f.filename);
                    match soas.remove_file(&f.id) {
                        Ok(true) => println!("   ✅ '{}' eliminado del índice (archivo real no se tocó).", f.filename),
                        Ok(false) => println!("   ⚠️  No se encontró el archivo."),
                        Err(e) => println!("   ❌ Error: {}", e),
                    }
                } else {
                    println!("   Número inválido. Usa 'archivos' para ver la lista.");
                }
            } else {
                println!("   Uso: borrar <número>");
            }
            println!();
            continue;
        }

        if query == "ifo" || query.starts_with("ifo ") {
            println!("   ¿Quisiste decir 'info <número>'? Usa TAB para autocompletar.");
            println!();
            continue;
        }

        let start = Instant::now();
        match soas.search(&query).await {
            Ok(results) => {
                let elapsed = start.elapsed();
                if results.is_empty() {
                    println!("   Sin resultados.");
                } else {
                    println!(
                        "   {} resultados en {:.2}s:\n",
                        results.len(),
                        elapsed.as_secs_f64()
                    );
                    for (i, r) in results.iter().enumerate().take(10) {
                        let icon = match r.file.extension.as_str() {
                            "pdf" => "📕",
                            "docx" | "doc" => "📘",
                            "xlsx" | "csv" => "📊",
                            "png" | "jpg" | "jpeg" | "svg" | "webp" => "🖼️",
                            "zip" | "tar" | "gz" => "📦",
                            _ => "📄",
                        };

                        let virtual_name = r
                            .virtual_info
                            .as_ref()
                            .map(|v| format!(" → \"{}\"", v.virtual_name))
                            .unwrap_or_default();

                        println!(
                            "   {}. {} {} (score: {:.2}){}",
                            i + 1,
                            icon,
                            r.file.filename,
                            r.score,
                            virtual_name,
                        );

                        // Mostrar summary o snippet como preview del contenido
                        let preview_text = r.file.metadata.summary.as_deref()
                            .filter(|s| !s.is_empty())
                            .or_else(|| {
                                if !r.snippet.is_empty() { Some(r.snippet.as_str()) } else { None }
                            });

                        if let Some(text) = preview_text {
                            let preview: String = text.chars().take(140).collect();
                            let preview = preview.replace('\n', " ");
                            println!("      └─ {}", preview.trim());
                        }
                    }
                }
            }
            Err(e) => {
                println!("   ❌ Error: {}", e);
            }
        }
        println!();
    }

    // Guardar estado
    soas.save()?;
    println!("\n👋 ¡Hasta luego!");

    Ok(())
}
