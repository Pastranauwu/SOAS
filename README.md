# SOAS Core - Sistema de Organización Autónoma Supervisada

SOAS Core es el motor de un **Sistema Inteligente de Búsqueda y Organización de Archivos** diseñado para indexar, analizar y recuperar información de tu sistema de archivos local de manera eficiente.

Este repositorio contiene el **núcleo (backend)** escrito en **Rust**, que proporciona la lógica de indexación, análisis semántico y gestión de datos. El objetivo a futuro es utilizar este core como base para construir un **Gestor de Archivos con IA** completo.

## 🚀 Características

*   **Búsqueda Semántica:** Utiliza modelos de IA (vía Ollama) para entender el contexto de tus búsquedas, no solo palabras clave.
*   **Indexación Rápida:** Motor de indexación escrito en Rust con `notify` para monitoreo en tiempo real.
*   **Soporte Multiformato:** Extrae y analiza contenido de PDF, CSV, DOCX, Imágenes y texto plano.
*   **Multiplataforma:** Funciona en Linux y Windows.
*   **Privacidad Local:** Todo el procesamiento y almacenamiento (SQLite) se realiza en tu máquina.

## 🧱 Arquitectura

Este componente (`soas-core`) es el cerebro del sistema y maneja:

*   **Watcher:** Monitorización de cambios en el sistema de archivos.
*   **Extractores:** Parsing de diferentes tipos de archivos y metadatos.
*   **Embeddings:** Generación de vectores semánticos con Ollama.
*   **Storage:** Almacenamiento eficiente en SQLite.
*   **CLI:** Interfaz de línea de comandos para gestión, búsquedas y depuración.

---

## 🛠️ Requisitos Previos

Para ejecutar y compilar este proyecto, necesitas:

### General
*   **Ollama:** Necesario para las funciones de IA. [Descargar Ollama](https://ollama.com/) y asegúrate de tener descargado un modelo de embeddings (ej. `nomic-embed-text` o similar compatible).

### Linux
*   **Rust:** `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
*   **Dependencias del sistema:**
    ```bash
    sudo apt update
    sudo apt install build-essential libssl-dev pkg-config libsqlite3-dev cmake clang
    ```

### Windows
*   **Rust:** Descargar e instalar `rustup-init.exe` desde [rust-lang.org](https://www.rust-lang.org/tools/install).
*   **Visual Studio Build Tools:** Necesario para compilar dependencias de C++. Asegúrate de incluir "Desarrollo para el escritorio con C++".

---

## ⚡ Guía de Inicio Rápido

### Compilar y Ejecutar

Asegúrate de tener Ollama corriendo antes de iniciar.

```bash
# Compilar el proyecto en modo release
cargo build --release

# Ejecutar el CLI
cargo run --bin soas-cli
```

**Comandos del CLI:**
Una vez dentro del CLI de `soas`, puedes interactuar con el sistema:
*   `help`: Ver todos los comandos disponibles.
*   `add <path>`: Agregar un directorio para monitorear.
*   `rescan`: Escanear directorios configurados.
*   `stats`: Ver estadísticas del índice.
*   `search <query>`: Realizar una búsqueda semántica.

---

## 🤝 Contribuciones

¡Este proyecto está abierto a contribuciones! Actualmente estamos enfocados en robustecer el núcleo para soportar la futura interfaz gráfica.

1.  **Fork** el repositorio.
2.  Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`).
3.  Haz tus cambios y **commit** (`git commit -m 'Agrega nueva funcionalidad'`).
4.  Haz **push** a la rama (`git push origin feature/nueva-funcionalidad`).
5.  Abre un **Pull Request**.

### Áreas de interés:
*   Soporte para más formatos de archivo en `src/content/`.
*   Optimización de consultas SQL y generación de embeddings.
*   Mejoras en el CLI y manejo de errores.

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.