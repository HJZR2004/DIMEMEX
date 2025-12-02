import pandas as pd
import json
import os
import numpy as np


BASE_DIR = "../data" 
TRAIN_FOLDER = "train"

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

def directorios(task_name):
    
    folder_path = os.path.join(BASE_DIR, TRAIN_FOLDER)
    

    # cargamos el json para obtener los nombres de archivo
    json_path = os.path.join(folder_path, f"{TRAIN_FOLDER}_data.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        data_json = json.load(f)
    
    df = pd.DataFrame(data_json)
    
    # cargamos las etiquetas
    csv_conf = TASKS[task_name]
    labels_path = os.path.join(folder_path, csv_conf["file"])
    df_labels = pd.read_csv(labels_path, header=None)
    
    # pasamos de one-hot encoding a etiquetas numéricas
    df['label'] = df_labels.values.argmax(axis=1)
    
    # Usamos os.path.join para asegurar compatibilidad entre sistemas operativos
    images_prefix = os.path.join("data", TRAIN_FOLDER, "images")
    df['path'] = df['MEME-ID'].apply(lambda x: os.path.join(images_prefix, x))
    
    final_df = df[['path', 'label']]
    
    # Guardamos el DataFrame final en un nuevo archivo CSV
    output_dir = os.path.join(BASE_DIR, "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"dataset_{task_name}.csv")
    final_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    directorios("simple")
    directorios("complex")