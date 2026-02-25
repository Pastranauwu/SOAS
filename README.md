# SOAS Core - Sistema de Organización Autónoma Supervisada

SOAS Core es el motor de un **Gestor de Archivos con IA** diseñado para indexar, analizar, buscar y organizar archivos del sistema de forma local y privada.

Escrito en **Rust**, expone una capa FFI compatible con C que Flutter (vía `dart:ffi`) consume directamente como biblioteca nativa.

---

## 🚀 Características

| Área | Descripción |
|---|---|
| **Búsqueda semántica** | Vectores de embeddings generados con Ollama para búsquedas por significado |
| **Spotlight search** | Búsqueda rápida (< 5 ms) en nombres de archivos indexados con ranking de relevancia |
| **Gestor de archivos** | Navega el sistema de archivos real, obtén info de rutas, abre y revela archivos |
| **Indexación multiformato** | PDF, DOCX, CSV, imágenes (visión), texto plano |
| **Organización virtual** | Categorías y archivos virtuales sin mover los archivos físicos |
| **Monitoreo de carpetas** | Seguimiento de carpetas con configuración por exclusiones y recursividad |
| **Almacenamiento local** | SQLite + FTS5 + vector store en memoria con persistencia a disco |
| **Privacidad total** | Todo el procesamiento ocurre en la máquina del usuario |

---

## 🧱 Arquitectura

```
soas-core/
├── ffi.rs          ← Capa FFI: todas las funciones exportadas a Flutter
├── config.rs       ← Configuración (rutas, modelos, límites)
├── models.rs       ← Structs de dominio (IndexedFile, Category, SearchResult…)
├── error.rs        ← Tipos de error unificados
├── storage/
│   └── sqlite.rs   ← Persistencia: SQLite + FTS5 + índices
├── vector_store/
│   └── memory.rs   ← Vector store en memoria (similitud coseno)
├── embeddings/
│   └── ollama.rs   ← Cliente HTTP para Ollama (embeddings + visión)
├── indexer/
│   └── pipeline.rs ← Pipeline: hash → extracción → embedding → clasificación
├── content/        ← Extractores por formato (pdf, docx, csv, imagen, texto)
├── search/
│   └── engine.rs   ← Motor de búsqueda semántica (híbrida FTS + vector)
├── virtual_fs/
│   └── manager.rs  ← Gestión de categorías y archivos virtuales
└── bin/cli.rs      ← CLI para desarrollo y depuración
```

---

## 🛠️ Requisitos Previos

### General
- **Ollama** corriendo localmente: [ollama.com](https://ollama.com/)
  - Modelo de embeddings: `nomic-embed-text` (configurable en `config.json`)
  - Modelo de visión (opcional para imágenes): `llava` o similar

### Linux
```bash
sudo apt install build-essential libssl-dev pkg-config libsqlite3-dev cmake clang
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Windows
- **Rust** desde [rust-lang.org](https://www.rust-lang.org/tools/install)
- **Visual Studio Build Tools** con "Desarrollo para el escritorio con C++"

---

## ⚡ Compilación

```bash
# Desarrollo
cargo build

# Release (para distribuir con Flutter)
cargo build --release

# CLI de desarrollo
cargo run --bin soas-cli
```

### Comandos del CLI
```
help           Ver todos los comandos
add <path>     Agregar carpeta al monitoreo
scan           Escanear todas las carpetas
scan <id>      Escanear carpeta específica
list           Listar carpetas monitoreadas
search <query> Búsqueda semántica
spotlight <q>  Búsqueda rápida por nombre
files          Listar archivos indexados
stats          Ver estadísticas del sistema
health         Verificar estado de Ollama
```

---

## 🔌 Integración con Flutter (FFI)

El core compila como biblioteca dinámica (`libsoas_core.so` / `soas_core.dll`) que Flutter carga mediante `dart:ffi`. Todas las funciones siguen la misma convención:

- **Input:** punteros `*const c_char` (strings C UTF-8)
- **Output:** `*mut c_char` con JSON `{ "success": true, "data": ... }` o `{ "success": false, "error": "..." }`
- **Liberación de memoria:** llamar `soas_free_string(ptr)` después de leer cada resultado

### Ciclo de vida

```dart
// 1. Inicializar (una sola vez al arrancar la app)
soas_init(null)  // config por defecto

// 2. Usar funciones…

// 3. Al cerrar la app
soas_destroy()
```

---

## 📋 Referencia Completa de la API FFI

### Ciclo de vida

| Función | Parámetros | Descripción |
|---|---|---|
| `soas_init(config_path)` | `*const c_char` (nullable) | Inicializa motor, SQLite, vector store y runtime Tokio |
| `soas_destroy()` | — | Guarda vectores pendientes y libera recursos |
| `soas_free_string(ptr)` | `*mut c_char` | **Obligatorio** llamar después de leer cualquier resultado |

---

### Exploración del sistema de archivos (navegación real)

Estas funciones operan sobre el sistema de archivos del SO, no sobre el índice.

| Función | Parámetros | Retorna |
|---|---|---|
| `soas_browse_directory(path)` | ruta absoluta | Listado de entradas con nombre, tipo, tamaño, modificación e `is_indexed` |
| `soas_get_home_dir()` | — | Ruta home del usuario (`/home/user`) |
| `soas_get_special_dirs()` | — | Array de `{key, path}` para Documentos, Descargas, Escritorio, etc. |
| `soas_path_info(path)` | ruta | Metadatos completos: `is_dir`, `size`, `created`, `modified`, `extension`, `parent` |
| `soas_path_exists(path)` | ruta | `true` / `false` |

**Ejemplo de respuesta `soas_browse_directory`:**
```json
{
  "path": "/home/user/Documents",
  "parent": "/home/user",
  "total": 42,
  "entries": [
    {
      "name": "Facturas",
      "path": "/home/user/Documents/Facturas",
      "is_dir": true,
      "is_symlink": false,
      "size": 0,
      "modified": "2025-12-01T10:00:00Z",
      "extension": null,
      "is_indexed": false
    },
    {
      "name": "reporte.pdf",
      "path": "/home/user/Documents/reporte.pdf",
      "is_dir": false,
      "is_symlink": false,
      "size": 204800,
      "modified": "2025-11-03T10:22:00Z",
      "extension": "pdf",
      "is_indexed": true
    }
  ]
}
```

---

### Búsqueda rápida tipo Spotlight

| Función | Parámetros | Descripción |
|---|---|---|
| `soas_spotlight_search(query, limit)` | texto parcial, máx resultados | Búsqueda en nombres de archivos indexados. Típicamente < 5 ms |

**Ranking de relevancia:**

| `match_rank` | Criterio | Ejemplo para `"repo"` |
|---|---|---|
| `1` | Nombre exacto (sin extensión) | `repo.txt` |
| `2` | Nombre empieza por la consulta | `reporte.pdf` |
| `3` | Alguna palabra del nombre empieza por la consulta | `Mi reporte 2025.docx` |
| `4` | Nombre contiene la consulta en cualquier posición | `balance_repo_anual.xlsx` |
| `5` | Solo la ruta contiene la consulta | `/proyectos/repo_viejo/cfg.json` |

**Ejemplo de respuesta:**
```json
{
  "query": "reporte",
  "total": 3,
  "elapsed_ms": 2,
  "results": [
    {
      "id": "abc-123",
      "filename": "reporte_2025.pdf",
      "path": "/home/user/Docs/reporte_2025.pdf",
      "extension": "pdf",
      "mime_type": "application/pdf",
      "size": 204800,
      "modified": "2025-11-03T10:22:00Z",
      "match_rank": 2,
      "is_dir": false
    }
  ]
}
```

---

### Búsqueda semántica (IA)

| Función | Parámetros | Descripción |
|---|---|---|
| `soas_quick_search(query_text)` | texto libre | Búsqueda semántica con configuración por defecto (límite 10) |
| `soas_search(query_json)` | JSON `SearchQuery` | Búsqueda semántica completa con filtros |

**JSON de entrada para `soas_search`:**
```json
{
  "text": "facturas de electricidad del año pasado",
  "limit": 20,
  "min_score": 0.4,
  "filters": {
    "extensions": ["pdf"],
    "date_from": "2024-01-01T00:00:00Z",
    "path_contains": "/Facturas"
  }
}
```

---

### Consulta de archivos indexados

| Función | Parámetros | Descripción |
|---|---|---|
| `soas_list_files(filter_json)` | JSON de filtros | Listado paginado con filtros opcionales |
| `soas_get_file(file_id)` | ID del archivo | Obtiene un archivo por su ID |
| `soas_get_files_in_path(path, limit, offset)` | ruta, límite, offset | Archivos indexados bajo una ruta |
| `soas_get_recent_files(limit)` | número | Archivos modificados más recientemente |
| `soas_get_files_by_extension(extensions_json, limit, offset)` | `["pdf","docx"]`, límite, offset | Filtra por extensiones |
| `soas_get_distinct_extensions()` | — | Lista todas las extensiones presentes en el índice |

**JSON de entrada para `soas_list_files`:**
```json
{
  "limit": 50,
  "offset": 0,
  "extension": "pdf",
  "path_prefix": "/home/user/Documents",
  "sort_by": "modified_at"
}
```
> `sort_by` acepta: `modified_at` (defecto), `name`, `size`, `created_at`, `indexed_at`

---

### Operaciones sobre archivos

| Función | Parámetros | Descripción |
|---|---|---|
| `soas_open_file(path)` | ruta | Abre con la app predeterminada del SO (`xdg-open` / `open`) |
| `soas_reveal_in_folder(path)` | ruta | Abre el directorio padre en el gestor de archivos del SO |
| `soas_delete_file_from_index(file_id)` | ID | Elimina del índice (el archivo físico no se toca) |

---

### Carpetas monitoreadas

| Función | Parámetros | Descripción |
|---|---|---|
| `soas_add_folder(path, name)` | ruta, nombre visible | Agrega carpeta al monitoreo |
| `soas_list_folders()` | — | Lista todas las carpetas monitoreadas |
| `soas_remove_folder(folder_id)` | ID | Elimina carpeta del monitoreo |
| `soas_update_folder(folder_id, update_json)` | ID, JSON de cambios | Edita `name`, `recursive`, `active`, `exclude_patterns` |

**JSON de entrada para `soas_update_folder`:**
```json
{
  "name": "Mis Documentos",
  "recursive": true,
  "active": true,
  "exclude_patterns": [".git", "node_modules", "target"]
}
```

---

### Indexación

| Función | Parámetros | Descripción |
|---|---|---|
| `soas_scan_all()` | — | Escanea e indexa todas las carpetas activas |
| `soas_scan_folder(folder_id)` | ID de carpeta | Escanea una carpeta específica |
| `soas_reindex_file(path)` | ruta absoluta | Re-indexa un único archivo (extrae contenido + genera embedding) |

---

### Organización virtual (Categorías)

| Función | Parámetros | Descripción |
|---|---|---|
| `soas_get_categories()` | — | Árbol completo de categorías |
| `soas_create_category(name, description, parent_id)` | nombre, desc, ID padre (nullable) | Crea categoría jerárquica |
| `soas_assign_file(file_id, category_id, virtual_name)` | IDs + nombre virtual | Asigna archivo a categoría con nombre personalizado |
| `soas_get_category_files(category_id)` | ID | Archivos de una categoría con su info virtual |

---

### Sistema

| Función | Parámetros | Descripción |
|---|---|---|
| `soas_get_stats()` | — | Total de archivos, embebidos, categorías, tamaño de BD |
| `soas_health_check()` | — | Verifica conectividad con Ollama y estado de los almacenes |

---

## 🤝 Contribuciones

1. Fork → rama (`git checkout -b feature/nombre`) → commit → push → Pull Request

**Áreas de interés:**
- Nuevos extractores de contenido en `src/content/`
- Optimización de consultas SQL
- Soporte de más plataformas (iOS, Android) en la capa FFI
- Watcher en tiempo real conectado a la UI

## 📄 Licencia

MIT. Consulta `LICENSE` para más detalles.
