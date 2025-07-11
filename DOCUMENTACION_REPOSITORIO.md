# Documentación del Repositorio WLASL

## 1. Introducción

Este repositorio contiene el dataset WLASL (Word-Level American Sign Language) y el código para entrenar y evaluar modelos de reconocimiento de lenguaje de señas a nivel de palabra. El objetivo principal es proporcionar recursos para la investigación y desarrollo en el campo del reconocimiento automático del lenguaje de señas americano (ASL).

El repositorio está estructurado para facilitar la descarga de datos, el preprocesamiento de videos y el entrenamiento/evaluación de dos arquitecturas de modelos principales: I3D (Inflated 3D ConvNet) y Pose-TGCN (Temporal Graph Convolutional Network).

## 2. Estructura del Repositorio

El repositorio tiene la siguiente estructura principal:

```
.
├── README.md               # Documentación principal del repositorio y dataset
├── AGENTS.md               # (Asumido, si existiera, para instrucciones a agentes AI)
├── start_kit/              # Scripts y datos para descargar y preprocesar videos
│   ├── WLASL_v0.3.json     # Metadatos del dataset (glosas, URLs, splits, etc.)
│   ├── video_downloader.py # Script para descargar videos
│   ├── preprocess.py       # Script para preprocesar videos (convertir, segmentar)
│   ├── data_reader.py      # Script ejemplo para leer WLASL_v0.3.json
│   ├── find_missing.py     # Script para encontrar videos faltantes
│   ├── C-UDA-1.0.pdf       # Acuerdo de uso de datos
│   └── scripts/
│       └── swf2mp4.sh      # Script auxiliar para conversión de video
├── code/                   # Código para los modelos de reconocimiento
│   ├── I3D/                # Modelo I3D (Inflated 3D ConvNet)
│   │   ├── train_i3d.py    # Script de entrenamiento para I3D
│   │   ├── test_i3d.py     # Script de prueba para I3D
│   │   ├── pytorch_i3d.py  # Implementación del modelo I3D
│   │   ├── datasets/       # Clases de Dataset para I3D
│   │   ├── configfiles/    # Archivos de configuración (.ini) para I3D
│   │   └── preprocess/     # JSONs de definición de subconjuntos para I3D
│   └── TGCN/               # Modelo Pose-TGCN
│       ├── train_tgcn.py   # Script de entrenamiento para TGCN
│       ├── test_tgcn.py    # Script de prueba para TGCN
│       ├── tgcn_model.py   # Implementación del modelo TGCN
│       ├── sign_dataset.py # Clase de Dataset para TGCN
│       └── configs/        # Archivos de configuración (.ini) para TGCN
└── ... (otros archivos de configuración y documentación)
```

### Componentes Principales:

*   **`README.md` (raíz):**
    *   Visión general del dataset WLASL.
    *   Instrucciones para la descarga y preprocesamiento de videos.
    *   Descripción del formato del archivo `WLASL_v0.3.json`.
    *   Instrucciones básicas para entrenar y probar los modelos I3D y Pose-TGCN.
    *   Información de citación y licencia.

*   **`start_kit/`:**
    *   **`WLASL_v0.3.json`**: Este es el archivo JSON central que contiene toda la metainformación del dataset. Cada entrada representa una "glosa" (palabra en ASL) y contiene una lista de "instancias" (videos de ejemplo para esa glosa). Para cada instancia, se provee:
        *   `gloss`: La palabra/seña que se está representando.
        *   `bbox`: Coordenadas del bounding box del signante (si está disponible).
        *   `fps`: Frames por segundo del video original (usualmente 25).
        *   `frame_start`, `frame_end`: Frames de inicio y fin de la seña dentro del video original. Es crucial para segmentar la seña específica. Un `frame_end` de -1 indica que la seña va hasta el final del video.
        *   `instance_id`: ID único de la instancia dentro de su clase/glosa.
        *   `signer_id`: ID del signante.
        *   `source`: Fuente del video (e.g., YouTube, ASLPro).
        *   `split`: Indica a qué conjunto de datos pertenece la instancia (`train`, `val`, `test`).
        *   `url`: URL para descargar el video.
        *   `video_id`: ID único del video.
    *   **`video_downloader.py`**: Script Python para descargar los videos listados en `WLASL_v0.3.json`. Utiliza `yt-dlp` (recomendado) o `youtube-dl` para videos de YouTube y métodos específicos para otras fuentes como ASLPro. Los videos descargados se guardan por defecto en `raw_videos/`.
    *   **`preprocess.py`**: Script Python que toma los videos crudos descargados por `video_downloader.py` y realiza dos tareas principales:
        1.  Conversión de formatos: Convierte videos (e.g., `.swf`, `.mkv`) a formato `.mp4` usando el script `scripts/swf2mp4.sh`. Los videos convertidos se guardan en `raw_videos_mp4/`.
        2.  Segmentación de instancias: Extrae los segmentos de video específicos para cada seña utilizando `frame_start` y `frame_end` del `WLASL_v0.3.json`. Los videos segmentados (instancias de señas individuales) se guardan por defecto en `videos/`.
    *   **`data_reader.py`**: Un script de ejemplo que muestra cómo cargar y leer el archivo `WLASL_v0.3.json` para obtener estadísticas básicas del dataset.
    *   **`find_missing.py`**: Ayuda a identificar qué videos no pudieron ser descargados, generando un archivo `missing.txt`.
    *   **`C-UDA-1.0.pdf`**: El "Computational Use of Data Agreement" que se debe leer y aceptar antes de usar el dataset.
    *   **`scripts/swf2mp4.sh`**: Un script de shell que utiliza `ffmpeg` para convertir archivos SWF a MP4. Es invocado por `preprocess.py`.

*   **`code/`:**
    *   **`I3D/`**: Implementación del modelo "Inflated 3D ConvNet", una arquitectura popular para reconocimiento de acciones en video.
        *   `train_i3d.py`: Script principal para entrenar el modelo I3D. Maneja la carga de datos (usando `datasets/nslt_dataset.py`), transformaciones de video, inicialización del modelo (con opción de cargar pesos pre-entrenados en Kinetics o ImageNet), y el bucle de entrenamiento/validación. Utiliza archivos de configuración (`.ini` en `configfiles/`) para parámetros como learning rate, batch size, etc., y JSONs específicos del dataset (en `preprocess/`, e.g., `nslt_2000.json`) que definen los subconjuntos de datos (WLASL100, WLASL300, WLASL1000, WLASL2000). Estos JSON parecen ser versiones preprocesadas o filtradas del `WLASL_v0.3.json` principal, adaptadas para el formato que espera el script de entrenamiento.
        *   `test_i3d.py`: Script para evaluar un modelo I3D previamente entrenado. Carga el dataset de prueba y el modelo guardado, y calcula métricas de precisión (Top-1, Top-5, Top-10).
        *   `pytorch_i3d.py`: Contiene la definición de la arquitectura del modelo InceptionI3d en PyTorch.
        *   `datasets/nslt_dataset.py` y `nslt_dataset_all.py`: Clases de PyTorch Dataset para cargar los videos y etiquetas para el entrenamiento y la prueba del modelo I3D.
        *   `configfiles/*.ini`: Archivos de configuración para diferentes subconjuntos del dataset (ASL100, ASL300, ASL1000, ASL2000), especificando hiperparámetros de entrenamiento.
        *   `preprocess/*.json`: Archivos JSON que definen las muestras para cada subconjunto del dataset (e.g., `nslt_100.json`, `nslt_1000.json`, `nslt_2000.json`). Estos archivos listan las instancias de video y sus etiquetas correspondientes para entrenamiento y prueba.
        *   `preprocess/wlasl_class_list.txt`: Un archivo de texto que lista los nombres de las clases (glosas).
    *   **`TGCN/`**: Implementación del modelo "Pose-Temporal Graph Convolutional Network". Este enfoque primero extrae características de pose (keypoints del esqueleto) de los videos y luego aplica GCNs para aprender las relaciones espaciales y temporales entre las poses.
        *   `train_tgcn.py`: Script para entrenar el modelo Pose-TGCN. Requiere datos de pose pre-extraídos y archivos de "split" específicos (mencionados en el `README.md` principal como descargas separadas).
        *   `test_tgcn.py`: Script para evaluar un modelo Pose-TGCN entrenado.
        *   `tgcn_model.py`: Define la arquitectura del modelo GCN con atención múltiple (`GCN_muti_att`).
        *   `sign_dataset.py`: Clase de PyTorch Dataset para cargar los datos de pose y etiquetas.
        *   `configs/*.ini`: Archivos de configuración para los diferentes subconjuntos del dataset para el modelo TGCN.
        *   El `README.md` principal indica que para usar este modelo, se deben descargar por separado los "splits file" y los "body keypoints".

## 3. Configuración del Entorno y Flujo de Trabajo

### 3.1. Descarga de Datos y Preprocesamiento (usando `start_kit/`)

1.  **Clonar el Repositorio:**
    ```bash
    git clone https://github.com/dxli94/WLASL.git
    cd WLASL
    ```

2.  **Instalar Dependencias:**
    *   Asegúrate de tener Python 3.x.
    *   Instala `yt-dlp` (recomendado) o `youtube-dl`. `yt-dlp` es un fork más actualizado.
        ```bash
        pip install yt-dlp
        ```
    *   Necesitarás `ffmpeg` para la conversión de video (usado por `scripts/swf2mp4.sh`). Instálalo según tu sistema operativo.
    *   Otras dependencias de Python pueden ser necesarias para los scripts (e.g., `opencv-python`). Puedes instalarlas con pip:
        ```bash
        pip install opencv-python numpy torch torchvision
        ```
        (Las dependencias exactas para los modelos I3D/TGCN están implícitas en sus scripts de entrenamiento).

3.  **Descargar Videos Crudos:**
    Navega al directorio `start_kit/` y ejecuta el script de descarga:
    ```bash
    cd start_kit
    python video_downloader.py
    ```
    Esto creará un directorio `raw_videos/` (por defecto) y descargará los videos de las URLs en `WLASL_v0.3.json`. El script maneja diferentes fuentes como YouTube y ASLPro. Se generará un archivo de log (e.g., `download_xxxx.log`).

4.  **Preprocesar Videos (Convertir y Segmentar):**
    Una vez descargados los videos crudos, ejecuta el script de preprocesamiento:
    ```bash
    python preprocess.py
    ```
    Este script:
    *   Primero llama a `bash scripts/swf2mp4.sh` para convertir archivos `.swf` (y otros formatos como `.mkv` según el script) a `.mp4`. Los resultados intermedios se guardan en `raw_videos_mp4/`.
    *   Luego, lee `WLASL_v0.3.json` y, para cada instancia, extrae el segmento de video relevante (usando `frame_start` y `frame_end`) del video MP4 correspondiente.
    *   Los videos segmentados finales (un video por cada instancia de seña) se guardan en el directorio `videos/` (por defecto). Estos son los videos que se usarán para el entrenamiento de modelos como I3D.

5.  **(Opcional) Encontrar Videos Faltantes:**
    Si algunos videos no se pudieron descargar, puedes generar una lista:
    ```bash
    python find_missing.py
    ```
    Esto crea `missing.txt`. El `README.md` principal ofrece un formulario de Google para solicitar videos faltantes o pre-procesados.

### 3.2. Entrenamiento y Prueba de Modelos (usando `code/`)

#### Modelo I3D (`code/I3D/`)

1.  **Preparar Datos:**
    *   Asegúrate de que los videos segmentados (generados en el paso anterior, usualmente en `WLASL/start_kit/videos/`) estén accesibles.
    *   El `README.md` sugiere crear un directorio `data/` en la raíz del proyecto y copiar/enlazar los videos allí, por ejemplo, para WLASL2000:
        ```bash
        cd WLASL  # Raíz del repo
        mkdir data
        cp -r start_kit/videos/ data/WLASL2000 # o ln -s start_kit/videos data/WLASL2000
        # Asegúrate que la estructura interna coincida con lo que espera el dataset loader.
        # El script train_i3d.py espera que los videos estén en root/{gloss}/{video_id}.mp4
        # Es posible que necesites reorganizar los videos de `start_kit/videos/` o modificar
        # `datasets/nslt_dataset.py` para que coincida con la estructura de `start_kit/videos/`
        # o con los JSONs en `code/I3D/preprocess/`.
        # Los JSON en `code/I3D/preprocess/` (e.g. `nslt_2000.json`) son los que realmente
        # guían al dataloader, especificando video_ids y sus clases.
        # El path `root` en `train_i3d.py` se combina con esta información.
        # Por defecto, `root` en `train_i3d.py` es `../../data/WLASL2000`.
        # Y los JSON en `code/I3D/preprocess/` listan paths como `videos/gloss/videoid.mp4`
        # Esto implica que la estructura debería ser `../../data/WLASL2000/videos/gloss/videoid.mp4`
        # Sin embargo, el `README.md` dice "put all the videos under data/" y luego "cp WLASL2000 -r data/"
        # que es confuso. La forma más segura es mirar `datasets/nslt_dataset.py` para ver cómo construye el path.
        # El script `nslt_dataset.py` construye el path como: `os.path.join(root, video_id)`.
        # Esto significa que los videos deben estar directamente bajo el directorio `root` especificado.
        # Si `root = '../../data/WLASL2000'`, entonces los videos deben estar en `../../data/WLASL2000/{video_id}.mp4`.
        # Los JSON en `code/I3D/preprocess/` contienen entradas como:
        # {"label": "book", "video_id": "07028.mp4", "bbox": [376, 66, 584, 478], "fps": 25, ...}
        # Así que el script de entrenamiento buscará `../../data/WLASL2000/07028.mp4`.
        # Por lo tanto, el directorio `videos/` de `start_kit` debe ser copiado o enlazado como `WLASL/data/WLASL2000`.
        # `cp -r start_kit/videos/ ../../data/WLASL2000` (si estás en `code/I3D`)
        # o desde la raíz: `cp -r start_kit/videos/ data/WLASL2000` (después de `mkdir data`)
        ```

2.  **Descargar Pesos Pre-entrenados (Opcional pero Recomendado):**
    *   Para I3D: El `README.md` proporciona un enlace para descargar pesos pre-entrenados en Kinetics (`weights/rgb_imagenet.pt` y `weights/flow_imagenet.pt`). Colócalos en la carpeta `code/I3D/weights/`.

3.  **Entrenar el Modelo I3D:**
    Navega al directorio `code/I3D/`:
    ```bash
    cd code/I3D
    ```
    Modifica `train_i3d.py` según sea necesario:
    *   Ajusta `root` para que apunte a tu directorio de datos (e.g., `../../data/WLASL2000`).
    *   Ajusta `train_split` para el archivo JSON del subconjunto deseado (e.g., `preprocess/nslt_2000.json` para WLASL2000).
    *   Ajusta `save_model` para el directorio donde se guardarán los checkpoints.
    *   Puedes especificar `weights` si tienes un checkpoint previo para continuar el entrenamiento.
    *   Elige el `mode` (`rgb` o `flow`). `flow` requiere precalcular optical flow.
    Ejecuta el script de entrenamiento:
    ```bash
    python train_i3d.py
    ```
    Los checkpoints se guardarán en el directorio especificado en `save_model` (por defecto `checkpoints/`).

4.  **Probar el Modelo I3D:**
    *   Descarga los pesos pre-entrenados específicos de WLASL si no los entrenaste tú mismo (el `README.md` proporciona un enlace). Colócalos en `code/I3D/archived/`.
    *   Modifica `test_i3d.py`:
        *   Ajusta `root` al directorio de datos.
        *   Ajusta `train_split` al JSON del subconjunto de prueba.
        *   Ajusta `weights` al path de tu modelo entrenado (e.g., `archived/asl2000/FINAL_nslt_2000_...pt`).
        *   Ajusta `num_classes` (e.g., 100, 300, 1000, 2000). El script `test_i3d.py` tiene esto hardcodeado en la sección `if __name__ == '__main__':`.
    Ejecuta el script de prueba:
    ```bash
    python test_i3d.py
    ```

#### Modelo Pose-TGCN (`code/TGCN/`)

1.  **Preparar Datos:**
    *   Según el `README.md` principal, necesitas descargar archivos de "splits" y "body keypoints" (poses) por separado. Los enlaces se proporcionan en el README.
    *   Descomprime estos archivos en `WLASL/data/`. Deberías tener `WLASL/data/splits/` y `WLASL/data/pose_per_individual_videos/`.
    *   Los archivos de split (e.g., `asl100.json`, `asl2000.json` dentro de `WLASL/data/splits/`) definen las muestras para entrenamiento y prueba, y apuntan a los archivos de pose.

2.  **Entrenar el Modelo TGCN:**
    Navega al directorio `code/TGCN/`:
    ```bash
    cd code/TGCN
    ```
    Modifica `train_tgcn.py` (principalmente en la sección `if __name__ == "__main__":`):
    *   Ajusta `root` para que apunte a la raíz de tu repositorio WLASL (e.g., `/path/to/WLASL`).
    *   Ajusta `subset` al subconjunto que deseas entrenar (e.g., `asl100`, `asl1000`, `asl2000`). Esto determinará qué archivo de split (`data/splits/{subset}.json`) y qué archivo de configuración (`code/TGCN/configs/{subset}.ini`) se utilizan.
    *   `pose_data_root` se construye a partir de `root`.
    Ejecuta el script de entrenamiento:
    ```bash
    python train_tgcn.py
    ```
    Los checkpoints se guardarán (por defecto) en `checkpoints/{subset}/`. Los logs se guardan en `output/`.

3.  **Probar el Modelo TGCN:**
    *   Descarga los modelos TGCN pre-entrenados (el `README.md` proporciona un enlace) y descomprímelos en `code/TGCN/archived/`.
    *   Modifica `test_tgcn.py` (principalmente en la sección `if __name__ == "__main__":`):
        *   Ajusta `root` a la raíz de tu repositorio WLASL.
        *   Ajusta `trained_on` al subconjunto con el que el modelo fue entrenado (e.g., `asl2000`). Esto afecta qué archivo de split, configuración y checkpoint se carga.
        *   Ajusta `checkpoint` al nombre del archivo del modelo entrenado (e.g., `ckpt.pth`).
    Ejecuta el script de prueba:
    ```bash
    python test_tgcn.py
    ```

## 4. Formatos de Datos Clave

*   **`WLASL_v0.3.json` (`start_kit/`):**
    *   El corazón del dataset. Es una lista de diccionarios, donde cada diccionario representa una glosa.
    *   Campos importantes por instancia: `url`, `video_id`, `frame_start`, `frame_end`, `split`, `gloss`.
    *   Usado por `video_downloader.py` y `preprocess.py`.

*   **JSONs de Preprocesamiento de I3D (`code/I3D/preprocess/`, e.g., `nslt_2000.json`):**
    *   Estos archivos JSON son utilizados directamente por los scripts de entrenamiento/prueba de I3D (`train_i3d.py`, `test_i3d.py`) a través de las clases `Dataset` en `code/I3D/datasets/`.
    *   Parecen ser una lista de diccionarios, donde cada diccionario representa una instancia de video individual (un segmento de seña).
    *   Contienen claves como `"label"` (la glosa), `"video_id"` (el nombre del archivo de video segmentado, e.g., `07028.mp4`).
    *   Estos archivos definen qué videos pertenecen a los conjuntos de entrenamiento y prueba para los subconjuntos específicos (WLASL100, WLASL2000, etc.).

*   **Archivos de Split de TGCN (`WLASL/data/splits/`, e.g., `asl100.json`):**
    *   Usados por los scripts de entrenamiento/prueba de TGCN.
    *   Contienen información para dividir los datos y apuntan a los datos de pose correspondientes. La estructura exacta necesitaría ser inspeccionada, pero típicamente contendrían identificadores de video/instancia y sus etiquetas de clase, divididos en conjuntos de entrenamiento, validación y prueba.

*   **Datos de Pose para TGCN (`WLASL/data/pose_per_individual_videos/`):**
    *   Estos son los keypoints del esqueleto extraídos de los videos. El formato exacto (e.g., archivos `.npy` o `.json` por video) dependerá de cómo fueron generados (el README no detalla este proceso de extracción, solo provee los datos pre-extraídos). `sign_dataset.py` en `code/TGCN/` se encarga de leer estos datos.

## 5. Consideraciones Adicionales

*   **Dependencias de Modelos:** Tanto I3D como TGCN dependen de PyTorch. Asegúrate de tener una versión compatible instalada, preferiblemente con soporte CUDA si tienes una GPU NVIDIA.
*   **Optical Flow para I3D (modo 'flow'):** Si planeas entrenar o usar el modelo I3D en modo `flow`, necesitarás precalcular el flujo óptico de los videos. El repositorio no incluye scripts para esto, por lo que tendrías que usar herramientas externas.
*   **Extracción de Pose para TGCN:** Si quisieras usar el modelo TGCN con tus propios datos o videos no incluidos en los keypoints pre-extraídos, necesitarías un pipeline para la extracción de pose 2D o 3D (e.g., usando OpenPose, MediaPipe, o similares).

Esta documentación proporciona una guía comprensiva para entender y utilizar el repositorio WLASL. Para detalles más específicos sobre las arquitecturas de los modelos o los procesos de carga de datos, se recomienda consultar directamente el código fuente relevante.
