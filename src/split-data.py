import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Configuracion de rutas y archivos
DATA_DIR = "../data/processed"
TEXT_FILE = "cleaned-ocr.csv"
SIMPLE_FILE = "dataset-simple.csv"
COMPLEX_FILE = "dataset-complex.csv"
OUTPUT_DIR = "../data/processed/splits"

def create_master_splits():
    # Cargar archivos CSV de texto y etiquetas
    df_text = pd.read_csv(os.path.join(DATA_DIR, TEXT_FILE))
    df_simple = pd.read_csv(os.path.join(DATA_DIR, SIMPLE_FILE))
    df_complex = pd.read_csv(os.path.join(DATA_DIR, COMPLEX_FILE))
    
    # Generar columna id basada en el nombre del archivo si no existe
    if 'id' not in df_simple.columns:
        df_simple['id'] = df_simple['path'].apply(os.path.basename)
    if 'id' not in df_complex.columns:
        df_complex['id'] = df_complex['path'].apply(os.path.basename)

    # Renombrar columnas de etiquetas para diferenciar tareas simple y compleja
    df_simple = df_simple.rename(columns={'label': 'label-simple'})
    df_complex = df_complex.rename(columns={'label': 'label-complex'})
    
    # Unir dataframes de etiquetas simples y complejas usando el id
    df_labels_merged = pd.merge(df_simple, df_complex[['id', 'label-complex']], on='id', how='inner')
    
    # Unir el resultado anterior con el texto limpio
    df_master = pd.merge(df_labels_merged, df_text[['id', 'text_clean']], on='id', how='inner')
    
    # Realizar particion estratificada basada en la etiqueta compleja
    # Separar conjunto de prueba del total
    df_temp, df_test = train_test_split(
        df_master, 
        test_size=0.15, 
        random_state=42, 
        stratify=df_master['label-complex'] 
    )
    
    # Separar conjuntos de entrenamiento y validacion del restante
    df_train, df_val = train_test_split(
        df_temp, 
        test_size=0.176, 
        random_state=42, 
        stratify=df_temp['label-complex']
    )
    
    # Crear directorio de salida y guardar los tres archivos CSV resultantes
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_train.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    df_val.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    df_test.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)
    
    print(f"Particion de datos finalizada y guardada en {OUTPUT_DIR}")

if __name__ == "__main__":
    create_master_splits()