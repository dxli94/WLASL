# Plan de Fine-tuning de Modelo de IA para Reconocimiento de Señas (ASL) con el Repositorio WLASL

## 0. Visión del Producto

Una aplicación móvil bidireccional que actúa como intérprete en tiempo real entre usuarios oyentes y personas Sordas:

*   **Modo Sign-to-Text (S2T):**
    *   El usuario apunta la cámara al interlocutor Sordo.
    *   La app graba o extrae información en tiempo real ➜ modelo Gemma-3N fino-ajustado ➜ texto español en pantalla y síntesis TTS opcional.
*   **Modo Text-to-Sign (T2S):**
    *   El usuario oyente escribe o dicta voz.
    *   El modelo genera la secuencia de glosas. (La generación visual de la seña a partir de glosas es un paso posterior y más complejo, no cubierto directamente por WLASL para la parte visual).

## 1. Análisis de Datos del Repositorio WLASL para Fine-tuning

El dataset WLASL (Word-Level American Sign Language) es un recurso valioso para el fine-tuning de un modelo de IA (como el hipotético Gemma-3N) para el reconocimiento de señas (modo S2T) y, en menor medida, para la traducción de texto a glosas (parte del modo T2S).

**Datos Relevantes en WLASL:**

*   **Videos de Señas:** El núcleo del dataset. Los scripts en `start_kit/` permiten descargar y preprocesar estos videos para obtener segmentos individuales de cada seña (`videos/` después de `preprocess.py`). Estos son los inputs visuales para el modelo S2T.
*   **Glosas (`gloss`):** Cada instancia de video está asociada a una "glosa", que es una etiqueta textual que representa la palabra o concepto de la seña en ASL (e.g., "book", "eat", "thank-you"). Estas glosas son las etiquetas objetivo (target labels) para el entrenamiento del modelo S2T. El archivo `WLASL_v0.3.json` y los JSON en `code/I3D/preprocess/` contienen esta información.
*   **Bounding Boxes (`bbox`):** Coordenadas (xmin, ymin, xmax, ymax) que delimitan al signante en el video. Pueden ser útiles para:
    *   Recortar el video para enfocar la atención del modelo en el signante.
    *   Como input adicional o para técnicas de aumento de datos.
*   **`frame_start` y `frame_end`:** Marcas temporales que definen el inicio y fin de una seña dentro de un video más largo. Cruciales para la segmentación realizada por `start_kit/preprocess.py`. El fine-tuning se realizará sobre estos segmentos ya aislados.
*   **Splits (`split`):** Divisiones predefinidas del dataset en conjuntos de `train`, `val`, y `test`. Es fundamental usar estas divisiones para una evaluación comparable y robusta del modelo.
*   **`video_id`:** Identificador único para cada video/segmento de seña.
*   **`signer_id`:** Identificador del signante. Podría ser útil para análisis de bias o para estrategias de división de datos más avanzadas (asegurarse de que los signantes en el conjunto de prueba no estén en el de entrenamiento).

**Limitaciones de WLASL para la Visión del Producto:**

*   **Nivel de Palabra (Word-Level):** WLASL está enfocado en señas aisladas. Para un S2T conversacional en tiempo real, se necesitaría un modelo que pueda manejar secuencias de señas (Continuous Sign Language Recognition - CSLR) y segmentarlas. WLASL puede ser un excelente punto de partida para entrenar un reconocedor de señas individuales que luego podría integrarse en un sistema CSLR más complejo.
*   **Texto en Español (S2T):** Las glosas en WLASL son generalmente palabras en inglés o transcripciones específicas de ASL. Para obtener "texto en español" como salida final, se necesitaría una capa de traducción adicional de glosa ASL -> español, o bien, re-etiquetar/mapear las glosas de WLASL a sus equivalentes directos en español si el objetivo es un vocabulario más limitado.
*   **Generación Visual para T2S:** WLASL no proporciona datos directos para entrenar la parte visual de un modelo Text-to-Sign (es decir, generar el video de la seña). Contiene los videos de humanos realizando las señas, que podrían servir como referencia o para modelos generativos más avanzados, pero no es su propósito principal. El T2S se enfocaría, usando WLASL, en la traducción de texto español a una secuencia de *glosas* de ASL.

## 2. Estrategia de Preparación de Datos para Fine-tuning

### 2.1. Descarga y Preprocesamiento Inicial con `start_kit`

1.  **Descargar Videos:** Utilizar `start_kit/video_downloader.py` para obtener todos los videos crudos.
2.  **Preprocesar Videos:** Utilizar `start_kit/preprocess.py` para:
    *   Convertir todos los videos a un formato estándar (MP4).
    *   Extraer los segmentos de video correspondientes a cada seña individual usando `frame_start` y `frame_end`. Estos videos segmentados (guardados en `videos/`) serán el input principal.

### 2.2. Preparación Específica para Fine-tuning (S2T)

1.  **Selección del Subconjunto de Datos:**
    *   Decidir qué subconjunto de WLASL se utilizará (WLASL100, 300, 1000, o WLASL2000). WLASL2000 ofrece la mayor cantidad de clases y datos, siendo ideal para un modelo más robusto, pero requiere más recursos computacionales.
    *   Utilizar los archivos JSON correspondientes en `code/I3D/preprocess/` (e.g., `nslt_2000.json`) o el `WLASL_v0.3.json` filtrado por el `split` deseado, para identificar las instancias de entrenamiento, validación y prueba.

2.  **Input del Modelo (Videos):**
    *   Los videos segmentados de `start_kit/videos/` serán los inputs.
    *   **Normalización/Redimensionamiento:** Los videos deben ser redimensionados a la resolución esperada por el modelo Gemma-3N (o su componente de visión). Considerar las transformaciones usadas en `code/I3D/videotransforms.py` como referencia (e.g., `RandomCrop(224)`, `CenterCrop(224)`).
    *   **Normalización de Píxeles:** Normalizar los valores de los píxeles (e.g., a `[0, 1]` o `[-1, 1]`).
    *   **Sampling de Frames:** Decidir cuántos frames por video se usarán y cómo se muestrearán (e.g., tomar N frames espaciados uniformemente, tomar todos los frames hasta un máximo). Esto dependerá de la arquitectura del modelo y de la duración variable de las señas. Los modelos I3D en el repositorio, por ejemplo, procesan secuencias de frames.

3.  **Output del Modelo (Etiquetas - Glosas):**
    *   Las `gloss` del archivo JSON serán las etiquetas.
    *   **Mapeo de Glosas a Índices:** Crear un vocabulario con todas las glosas únicas del subconjunto elegido y mapear cada glosa a un índice numérico entero. Esto es estándar para la mayoría de los frameworks de deep learning. El archivo `code/I3D/preprocess/wlasl_class_list.txt` ya provee una lista, que puede ser usada para este propósito. La capa de salida del modelo Gemma-3N tendrá tantas neuronas como clases (glosas) haya en el vocabulario.

4.  **Aumento de Datos (Data Augmentation):**
    *   Para mejorar la robustez y generalización del modelo, aplicar técnicas de aumento de datos a los videos:
        *   Flipping horizontal (si es semánticamente válido para ASL, algunas señas cambian de significado).
        *   Pequeñas rotaciones, escalados, traslaciones.
        *   Cambios de brillo, contraste.
        *   Time masking o frame dropping (simulando oclusiones o variaciones de velocidad).
    *   Referirse a `videotransforms.py` para ejemplos de transformaciones.

5.  **Creación de DataLoaders:**
    *   Implementar DataLoaders (e.g., en PyTorch o TensorFlow) que carguen eficientemente los videos y sus etiquetas correspondientes, aplicando las transformaciones y el aumento de datos en tiempo de ejecución. Los scripts en `code/I3D/datasets/` pueden servir de inspiración.

### 2.3. Preparación para T2S (Traducción Español a Glosas)

El fine-tuning directo para la generación visual de señas desde texto está más allá del alcance de WLASL. Sin embargo, WLASL puede ayudar en el paso intermedio de traducir texto en español a secuencias de glosas ASL.

1.  **Dataset Paralelo Español-Glosas:**
    *   Esto es el mayor desafío. WLASL proporciona (video, glosa). Para T2S (texto -> glosa), se necesitaría un dataset de pares (frase en español, secuencia de glosas ASL).
    *   **Posibles Enfoques (si no se tiene un dataset paralelo directo):**
        *   **Traducción de Glosas WLASL:** Si las glosas de WLASL (mayormente en inglés) se traducen al español, se podría intentar entrenar un modelo que aprenda a generar la glosa original en ASL a partir de la palabra en español. Esto sería a nivel de palabra, no de frase.
        *   **Uso de Subtítulos (si disponibles):** Si algunos videos originales de WLASL tuvieran subtítulos en español, se podría intentar alinearlos con las secuencias de glosas. Esto es complejo.
        *   **Creación Manual o Crowdsourcing:** Generar un nuevo dataset de frases en español y sus traducciones a secuencias de glosas ASL por expertos.
        *   **Simplificación:** Enfocarse en un vocabulario limitado donde haya una correspondencia uno-a-uno (o casi) entre palabras en español y glosas de WLASL.

2.  **Si se dispone de un dataset Español-Glosas:**
    *   **Tokenización:** Tokenizar tanto las frases en español como las secuencias de glosas.
    *   **Vocabularios:** Crear vocabularios para español y para las glosas.
    *   **Modelo Seq2Seq:** Un modelo como Gemma-3N (si tiene capacidades de secuencia a secuencia) podría ser entrenado para esta tarea de traducción.

## 3. Modelo de IA (Gemma-3N) y Estrategia de Fine-tuning

Asumimos que "Gemma-3N" es un modelo de IA pre-entrenado, posiblemente multimodal o con componentes adaptables para visión y secuencia.

### 3.1. Fine-tuning para Sign-to-Text (S2T)

1.  **Arquitectura del Modelo (Adaptación de Gemma-3N):**
    *   **Extractor de Características de Video:** Se necesita un componente que procese la secuencia de frames del video de la seña y extraiga características representativas. Esto podría ser:
        *   Un modelo 3D CNN (como el I3D del repositorio WLASL). Se podrían usar los pesos pre-entrenados en Kinetics o incluso los pesos entrenados en WLASL si se busca transferir conocimiento.
        *   Un Video Transformer (ViT) o similar.
        *   Si Gemma-3N tiene un "backbone" de visión, podría adaptarse para tomar secuencias de frames.
    *   **Agregación Temporal:** Las características de los frames deben ser agregadas o procesadas secuencialmente (e.g., con LSTMs, GRUs, Transformers, o atención temporal) para capturar la dinámica de la seña.
    *   **Clasificador:** Una capa final (fully connected) seguida de una función Softmax para predecir la probabilidad de cada glosa en el vocabulario.

2.  **Estrategia de Fine-tuning:**
    *   **Cargar Pesos Pre-entrenados:** Inicializar Gemma-3N (o sus componentes relevantes) con sus pesos pre-entrenados.
    *   **Adaptar la Capa de Salida:** Reemplazar la capa de clasificación final del modelo con una nueva capa que tenga el número de neuronas igual al tamaño del vocabulario de glosas de WLASL.
    *   **Congelación de Capas (Opcional):** Inicialmente, se pueden congelar las capas del "backbone" pre-entrenado y entrenar solo la capa de clasificación y algunas capas superiores del extractor de características. Luego, se pueden descongelar más capas (o todas) para un fine-tuning más profundo con un learning rate más bajo ("differential learning rates").
    *   **Función de Pérdida (Loss Function):** `CrossEntropyLoss` es adecuada para problemas de clasificación multiclase.
    *   **Optimizador:** Adam, AdamW, o SGD con momento.
    *   **Learning Rate Scheduler:** Reducir el learning rate durante el entrenamiento (e.g., `ReduceLROnPlateau` o `CosineAnnealingLR`).
    *   **Métricas de Entrenamiento y Validación:** Accuracy (Top-1, Top-5), Pérdida.

### 3.2. Fine-tuning/Entrenamiento para Text-to-Sign (Español a Glosas)

1.  **Arquitectura del Modelo (Adaptación de Gemma-3N):**
    *   Si Gemma-3N es un modelo de lenguaje grande (LLM) o tiene componentes de procesamiento de secuencias, podría adaptarse como un modelo de traducción neuronal (Encoder-Decoder).
    *   **Encoder:** Procesaría la secuencia de tokens de la frase en español.
    *   **Decoder:** Generaría la secuencia de tokens de glosas ASL.
    *   Se podría usar una arquitectura Transformer estándar para esto.

2.  **Estrategia de Entrenamiento:**
    *   Entrenar desde cero o fine-tuning si Gemma-3N ya tiene capacidades de traducción o secuencia a secuencia.
    *   **Función de Pérdida:** `CrossEntropyLoss` (a nivel de token de glosa).
    *   **Métricas:** BLEU, ROUGE (para evaluar la calidad de la secuencia de glosas generada en comparación con la referencia), Accuracy a nivel de token.

## 4. Métricas de Evaluación

### Para S2T (Reconocimiento de Señas):

*   **Accuracy (Top-1, Top-k):** Porcentaje de veces que la glosa correcta está entre las k predicciones principales del modelo. Es la métrica principal usada en el repositorio WLASL.
*   **Word Error Rate (WER):** Si se extiende a reconocimiento continuo, WER sería una métrica clave. Para señas aisladas, es similar al (1 - Accuracy).
*   **Matriz de Confusión:** Para entender qué señas se confunden con otras.
*   **Precisión, Recall, F1-score por clase:** Para evaluar el rendimiento en señas individuales, especialmente importante para clases con pocas muestras.

### Para T2S (Traducción a Glosas):

*   **BLEU Score:** Compara la similitud de la secuencia de glosas generada con una o más secuencias de referencia.
*   **ROUGE Score:** Similar a BLEU, enfocado en recall.
*   **METEOR, TER:** Otras métricas de evaluación de traducción automática.
*   **Exactitud de la Secuencia de Glosas:** Porcentaje de secuencias de glosas generadas que coinciden exactamente con la referencia.

## 5. Consideraciones Adicionales y Desafíos

*   **Escalabilidad:** WLASL2000 tiene 2000 clases. Un modelo robusto necesitará manejar un vocabulario aún mayor para una aplicación real.
*   **Variaciones en Señas:** ASL tiene variaciones regionales, de velocidad, y estilo entre signantes. El modelo debe ser robusto a estas variaciones. El campo `signer_id` podría usarse para asegurar diversidad de signantes en los conjuntos de datos.
*   **Señas Fuera de Vocabulario (OOV):** Cómo manejar señas que el modelo no ha visto durante el entrenamiento.
*   **Reconocimiento Continuo vs. Aislado:** La visión del producto implica reconocimiento en tiempo real, que generalmente requiere CSLR. WLASL es para señas aisladas. Se necesitaría un paso adicional para pasar de un modelo de señas aisladas a uno continuo.
*   **Datos para T2S (Visual):** Generar videos de señas realistas a partir de texto o glosas es un problema de investigación activo y muy complejo, que requiere diferentes tipos de modelos (GANs, VAEs, modelos de difusión) y datos (e.g., videos con captura de movimiento).
*   **Latencia en Tiempo Real:** Para una aplicación móvil, la inferencia del modelo debe ser rápida.
*   **Recursos Computacionales:** El fine-tuning de modelos grandes de visión y lenguaje requiere GPUs potentes y tiempo considerable.
*   **Ética e Inclusividad:**
    *   Asegurar que el dataset y el modelo no tengan sesgos hacia ciertos grupos de signantes.
    *   Involucrar a la comunidad Sorda en el desarrollo y prueba de la aplicación.

Este plan estratégico proporciona una hoja de ruta para aprovechar el repositorio WLASL en el desarrollo de la aplicación de interpretación de ASL, enfocándose principalmente en el componente Sign-to-Text (video a glosa).
