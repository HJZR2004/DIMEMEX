import pandas as pd
from sklearn.model_selection import train_test_split
import os


"""
Este codigo nos ayuda a dividir el dataset en tres particiones: entrenamiento, validación y prueba,
asegurando que la distribución de etiquetas se mantenga en cada partición.

estos archivos se guardaran en la carpeta data/processed/splits como train.csv, val.csv y test.csv
"""


# Configuracion de rutas
DATA_DIR = "../data/processed"
TEXT_FILE = "cleaned-ocr.csv"
LABEL_FILE = "dataset-simple.csv"

def create_partitions():
    # Cargar archivos CSV de texto y etiquetas
    df_text = pd.read_csv(os.path.join(DATA_DIR, TEXT_FILE))
    df_labels = pd.read_csv(os.path.join(DATA_DIR, LABEL_FILE))
    
    # Generar columna id extrayendo el nombre del archivo de la ruta
    df_labels['id'] = df_labels['path'].apply(os.path.basename)

    print("Columnas cargadas correctamente.")
    
    # Unir datasets usando el id y seleccionando solo columnas necesarias
    print("Uniendo informacion...")
    df_full = pd.merge(
        df_labels[['id', 'path', 'label']],
        df_text[['id', 'text_clean']],
        on='id', 
        how='inner'
    )
    
    print(f"Total de registros unidos: {len(df_full)}")
    
    # Separar conjunto de prueba (15%) del total
    df_temp, df_test = train_test_split(
        df_full, 
        test_size=0.15, 
        random_state=42, 
        stratify=df_full['label']
    )
    
    # Separar conjunto de validacion del restante (aprox 15% del original)
    df_train, df_val = train_test_split(
        df_temp, 
        test_size=0.176, 
        random_state=42, 
        stratify=df_temp['label']
    )
    
    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
    
    # Crear directorio de salida y guardar los tres archivos CSV
    output_dir = os.path.join(DATA_DIR, "splits")
    os.makedirs(output_dir, exist_ok=True)
    
    df_train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    df_test.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print(f"Archivos guardados en: {output_dir}")

if __name__ == "__main__":
    create_partitions()