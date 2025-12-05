# DIMEMEX - Detección de Memes Inapropiados de México

## Descripción del Proyecto

Este repositorio implementa un sistema de clasificación automática para detectar contenido inapropiado y discurso de odio en memes mexicanos. El proyecto surge de la necesidad de moderar contenido multimodal en redes sociales, donde los memes combinan texto e imagen para transmitir mensajes que pueden contener discriminación, odio o contenido nocivo disfrazado de humor.

El sistema utiliza técnicas de Visión por Computadora (OCR) para extraer texto de imágenes y Procesamiento de Lenguaje Natural (NLP) mediante el modelo BETO (BERT en español) para clasificar el contenido en categorías específicas de severidad y tipo de discurso.

## Arquitectura del Sistema

El pipeline de clasificación opera en dos etapas:

**Etapa 1: Extracción de Texto**
- Se utiliza EasyOCR para extraer el texto presente en las imágenes de memes
- El texto extraído pasa por un proceso de limpieza y normalización
- Se maneja texto en español e inglés, común en memes mexicanos

**Etapa 2: Clasificación con BETO**

El sistema implementa dos tareas de clasificación:

1. **Tarea Simple (3 clases)**
   - None: Contenido inofensivo
   - Inappropriate: Contenido inapropiado pero no odio
   - Hate: Discurso de odio

2. **Tarea Compleja (6 clases)**
   - None: Contenido inofensivo
   - Inappropriate: Inapropiado genérico
   - Sexism: Discriminación por género
   - Racism: Discriminación racial o étnica
   - Classicism: Discriminación por clase socioeconómica
   - Other: Otros tipos de discurso de odio

El modelo BETO utilizado es `dccuchile/bert-base-spanish-wwm-cased`, especializado en español y sensible a mayúsculas/minúsculas, lo que permite capturar sutilezas del lenguaje coloquial mexicano.

## Estructura del Proyecto

```
DIMEMEX/
├── app/
│   └── main.py                 # Interfaz web con Streamlit
├── computer-vision/
│   ├── ocr-analisy.py         # Análisis de calidad del OCR
│   └── ocr-benchmark.py       # Evaluación del OCR
├── data/                       # Datasets y resultados (requerido)
│   ├── processed/
│   │   ├── computer-vision/   # Resultados de benchmark OCR
│   │   ├── datasets/          # Datasets procesados
│   │   └── nlp/               # Datos limpios para entrenamiento
│   ├── train/                 # Datos de entrenamiento
│   ├── validation/            # Datos de validación
│   └── test/                  # Datos de prueba
├── models/                     # Modelos entrenados (requerido)
│   └── v4/
│       ├── beto_simple_v4.pth
│       └── beto_complex_v4.pth
├── natural-language-processing/
│   ├── exploracion.ipynb      # Análisis exploratorio del corpus
│   ├── limpieza.py            # Preprocesamiento de texto
│   ├── train.py               # Entrenamiento de modelos
│   └── test.py                # Evaluación de modelos
├── src/
│   ├── inference.py           # Pipeline de predicción
│   ├── nlp_utils.py           # Utilidades y arquitectura BETO
│   ├── utils.py               # Funciones auxiliares
│   └── split-data.py          # División de datasets
└── requeriments.txt           # Dependencias del proyecto
```

## Requisitos Previos

- Python 3.8 o superior
- CUDA (opcional, para entrenamiento con GPU)
- Al menos 4GB de RAM
- 2GB de espacio en disco para modelos y dependencias

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/HJZR2004/DIMEMEX.git
cd DIMEMEX
```

### 2. Crear entorno virtual

Es altamente recomendable usar un entorno virtual para evitar conflictos de dependencias:

```bash
python -m venv venv
```

Activar el entorno virtual:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requeriments.txt
```

Las dependencias principales incluyen:
- PyTorch (deep learning)
- Transformers (modelos BERT/BETO)
- EasyOCR (extracción de texto)
- Streamlit (interfaz web)
- Pandas, NumPy, Scikit-learn (procesamiento de datos)
- NLTK (procesamiento de lenguaje)

### 4. Descargar recursos de NLTK

Al ejecutar por primera vez, el sistema descargará automáticamente los recursos necesarios de NLTK (stopwords en español). Esto puede tomar unos minutos.

## Configuración de Carpetas Requeridas

**Importante:** El proyecto requiere que existan las carpetas `data/` y `models/` con su estructura completa para funcionar correctamente.

### Carpeta `data/`

Debe contener los datasets procesados y las imágenes de memes. La estructura mínima requerida es:

```
data/
├── processed/
│   ├── datasets/
│   │   ├── dataset-simple.csv
│   │   └── dataset-complex.csv
│   └── nlp/
│       └── complete-data.csv
├── train/images/
├── validation/images/
└── test/images/
```

Si no tienes acceso a los datos originales, contacta a los mantenedores del proyecto.

### Carpeta `models/`

Debe contener los modelos BETO entrenados. La estructura requerida es:

```
models/
└── v4/
    ├── beto_simple_v4.pth
    └── beto_complex_v4.pth
```

Los modelos `.pth` son archivos grandes (aproximadamente 400MB cada uno) que contienen los pesos del modelo entrenado. Estos archivos son esenciales para que el sistema de inferencia funcione.

**Nota:** Los modelos entrenados no están incluidos en el repositorio debido a su tamaño. Puedes:
- Entrenar tus propios modelos usando `natural-language-processing/train.py`
- Solicitar acceso a los modelos pre-entrenados contactando a los mantenedores

## Uso del Sistema

### Interfaz Web (Streamlit)

Para ejecutar la aplicación web de clasificación:

```bash
streamlit run app/main.py
```

Esto abrirá una interfaz en el navegador donde puedes:
- Subir una imagen de meme
- Seleccionar el tipo de clasificación (simple o compleja)
- Obtener la predicción del modelo con el nivel de confianza

### Entrenamiento de Modelos

Para entrenar modelos desde cero:

```bash
cd natural-language-processing
python train.py
```

El script te permitirá seleccionar:
- Tarea (simple o complex)
- Modo de entrenamiento (experimental o producción)
- Versión del modelo a guardar

Los modelos entrenados se guardarán en `models/vX/`.

### Evaluación de Modelos

Para evaluar un modelo en el conjunto de prueba:

```bash
cd natural-language-processing
python test.py
```

Esto generará métricas de rendimiento (accuracy, precision, recall, F1-score) y matrices de confusión.

### Análisis Exploratorio

Para explorar los datos y realizar análisis estadístico:

```bash
jupyter notebook natural-language-processing/exploracion.ipynb
```

El notebook incluye:
- Distribución de clases
- Análisis de frecuencias de palabras
- Ley de Zipf
- N-gramas y co-ocurrencias
- Análisis de redes semánticas

## Flujo de Trabajo Típico

1. **Preparación de datos:** Asegúrate de tener las imágenes y anotaciones en `data/`
2. **Limpieza de texto:** Ejecuta `limpieza.py` para preprocesar el texto extraído por OCR
3. **División de datos:** Usa `split-data.py` para crear conjuntos de train/val/test
4. **Entrenamiento:** Ejecuta `train.py` para entrenar los modelos
5. **Evaluación:** Usa `test.py` para validar el rendimiento
6. **Despliegue:** Ejecuta la aplicación Streamlit para uso interactivo

## Resultados y Rendimiento

Los modelos versión 4 (v4) representan la mejor iteración del proyecto, con:
- Balanceo de clases mediante pesos en la función de pérdida
- Regularización con dropout (20%)
- Fine-tuning de BETO en 4 épocas
- Learning rate de 2e-5

El rendimiento varía según la tarea:
- La tarea simple alcanza mayor accuracy debido a la simplicidad de las categorías
- La tarea compleja enfrenta desafíos por el desbalance de clases (pocas muestras de racismo y clasismo)

## Limitaciones Conocidas

- El sistema depende completamente de la calidad del OCR. Memes con texto distorsionado, en fuentes decorativas o con bajo contraste pueden tener errores de extracción
- El modelo solo analiza texto, no el contenido visual de la imagen
- El dataset de entrenamiento es relativamente pequeño (aproximadamente 3,000 memes)
- Algunas categorías están significativamente desbalanceadas
- El contexto cultural específico puede no generalizarse a otros países hispanohablantes

## Contribuciones

Este proyecto es académico y está en desarrollo activo. Las contribuciones son bienvenidas en las siguientes áreas:

- Mejora del preprocesamiento de imágenes para OCR
- Aumento de datos (data augmentation)
- Incorporación de features visuales además del texto
- Optimización de hiperparámetros
- Expansión del dataset con más ejemplos anotados

## Licencia

Este proyecto está bajo la licencia especificada en el archivo LICENSE.

## Contacto

Para preguntas sobre el proyecto, acceso a datos o modelos pre-entrenados, contacta a los mantenedores del repositorio.

## Reconocimientos

- Dataset DIMEMEX del challenge de detección de contenido inapropiado en memes mexicanos
- Modelo BETO por dccuchile
- Comunidad de Hugging Face y PyTorch por las herramientas de NLP