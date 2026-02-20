use soas_core::prelude::*;
use std::io::{self, Write};
use std::time::Instant;

const DESCARGAS: &str = "/home/eduardo/Descargas";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("soas_core=info".parse()?),
        )
        .init();

    println!("╔══════════════════════════════════════════╗");
    println!("║        SOAS - CLI de prueba              ║");
    println!("║   Sistema Inteligente de Búsqueda        ║");
    println!("╚══════════════════════════════════════════╝");
    println!();

    // 1. Inicializar SOAS
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

    let required = ["nomic-embed-text", "qwen2.5:3b", "qwen3-vl:2b"];
    for model in &required {
        let found = models.iter().any(|m| m.contains(model));
        if found {
            println!("   ✅ {}", model);
        } else {
            println!("   ⚠️  {} no encontrado (ollama pull {})", model, model);
        }
    }
    println!();

    // 4. Agregar carpeta de Descargas
    println!("📁 Configurando carpeta: {}", DESCARGAS);
    let folders = soas.list_folders()?;
    let already_added = folders.iter().any(|f| f.path.to_string_lossy() == DESCARGAS);

    if already_added {
        println!("   (ya estaba registrada)");
    } else {
        soas.add_folder(DESCARGAS, "Descargas").await?;
        println!("   ✅ Carpeta agregada");
    }
    println!();

    // 5. Crear categorías por defecto
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

    // 6. Escanear e indexar
    println!("🔍 Escaneando e indexando {}...", DESCARGAS);
    println!("   Solo se procesarán archivos nuevos o modificados.");
    println!();

    let start = Instant::now();
    let results = soas.scan_all().await?;
    let elapsed = start.elapsed();

    for result in &results {
        println!("   📊 Resultado del escaneo:");
        println!("      Nuevos:       {}", result.new_files);
        println!("      Actualizados: {}", result.updated_files);
        println!("      Eliminados:   {}", result.deleted_files);
        println!("      Errores:      {}", result.failed_files);
        println!("      Total:        {}", result.total_scanned);
    }
    println!("   ⏱️  Tiempo: {:.1}s", elapsed.as_secs_f64());
    println!();

    // 7. Estadísticas
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

    // 8. Loop interactivo de búsqueda
    println!("═══════════════════════════════════════════");
    println!("  Búsqueda interactiva");
    println!("  Busca en lenguaje natural. Ejemplos:");
    println!("   • \"foto de mi credencial INE\"");
    println!("   • \"informe de actividades de noviembre\"");
    println!("   • \"pdf sobre licitaciones del gobierno\"");
    println!("  Comandos especiales:");
    println!("   stats | cats | rescan | rebuild | reindex | reimages | salir");
    println!("   carpetas         = lista carpetas monitoreadas");
    println!("   agregar <ruta>   = agrega una carpeta para indexar");
    println!("   quitar <num>     = elimina carpeta y sus archivos del índice");
    println!("   archivos         = lista todos los archivos indexados");
    println!("   info <num>       = muestra detalles de un archivo");
    println!("   borrar <num>     = elimina un archivo del índice");
    println!("   🔄 rebuild       = regenera embeddings con formato mejorado");
    println!("   🖼️  reimages     = re-analiza solo imágenes (visión+LLM)");
    println!("═══════════════════════════════════════════");
    println!();

    loop {
        print!("🔎 Buscar: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let query = input.trim();

        if query.is_empty() {
            continue;
        }
        if query == "salir" || query == "exit" || query == "q" {
            break;
        }

        // Comandos especiales
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
            let results = soas.scan_all().await?;
            let elapsed = start.elapsed();
            for r in &results {
                println!("   Nuevos: {} | Actualizados: {} | ⏱️ {:.1}s",
                    r.new_files, r.updated_files, elapsed.as_secs_f64());
            }
            println!();
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
                        soas_core::prelude::IndexStatus::Indexed => "✅",
                        soas_core::prelude::IndexStatus::Failed(_) => "❌",
                        soas_core::prelude::IndexStatus::Pending => "⏳",
                        soas_core::prelude::IndexStatus::ContentExtracted => "📝",
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

        // Búsqueda semántica
        let start = Instant::now();
        match soas.search(query).await {
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

                        if !r.snippet.is_empty() {
                            let snippet: String = r.snippet.chars().take(120).collect();
                            let snippet = snippet.replace('\n', " ");
                            println!("      └─ {}", snippet.trim());
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
