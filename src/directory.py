import pandas as pd
import json
import os

# Configuracion de rutas dinamicas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../"))

# Directorios de entrada y salida
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "train")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "datasets")

TASKS = {
    "simple": {
        "file": "label_simple.csv",
        "cols": ["none", "inappropriate", "hate_speech"] 
    },
    "complex": {
        "file": "label_complex.csv",
        "cols": ["none", "inappropriate", "sexism", "racism", "classicism", "other"]
    }
}

def create_dataset_csv(task_name):
    # Validar existencia del directorio de entrada
    if not os.path.exists(INPUT_DIR):
        print(f"Error: No se encuentra el directorio {INPUT_DIR}")
        return

    # Cargar archivo JSON con metadatos
    json_path = os.path.join(INPUT_DIR, "train_data.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        data_json = json.load(f)
    
    df = pd.DataFrame(data_json)
    
    # Cargar archivo CSV con etiquetas
    csv_conf = TASKS[task_name]
    labels_path = os.path.join(INPUT_DIR, csv_conf["file"])
    df_labels = pd.read_csv(labels_path, header=None)
    
    # Convertir one-hot encoding a etiquetas numericas
    df['label'] = df_labels.values.argmax(axis=1)
    
    # Generar rutas relativas de las imagenes
    images_prefix = os.path.join("data", "train", "images")
    df['path'] = df['MEME-ID'].apply(lambda x: os.path.join(images_prefix, x))
    
    # Seleccionar columnas finales
    final_df = df[['path', 'text', 'description', 'label']]
    
    # Crear directorio de salida si no existe
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Guardar archivo CSV
    output_file = os.path.join(OUTPUT_DIR, f"dataset-{task_name}.csv")
    final_df.to_csv(output_file, index=False)
    print(f"Archivo generado: {output_file}")

if __name__ == "__main__":
    create_dataset_csv("simple")
    create_dataset_csv("complex")