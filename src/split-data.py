import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Configuracion de rutas dinamicas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../"))

# Directorios de entrada y salida
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed","nlp")
TEXT_FILE_PATH = os.path.join(DATA_PROCESSED_DIR, "cleaned-ocr.csv")
SIMPLE_FILE_PATH = os.path.join(DATA_PROCESSED_DIR, "datasets", "dataset-simple.csv")
COMPLEX_FILE_PATH = os.path.join(DATA_PROCESSED_DIR, "datasets", "dataset-complex.csv")
OUTPUT_DIR = os.path.join(DATA_PROCESSED_DIR, "splits")

def create_master_splits():
    # Validar existencia de archivos de entrada
    if not all(os.path.exists(p) for p in [TEXT_FILE_PATH, SIMPLE_FILE_PATH, COMPLEX_FILE_PATH]):
        print("Error: Faltan archivos de entrada en data/processed")
        return

    # Cargar archivos CSV
    df_text = pd.read_csv(TEXT_FILE_PATH)
    df_simple = pd.read_csv(SIMPLE_FILE_PATH)
    df_complex = pd.read_csv(COMPLEX_FILE_PATH)
    
    # Generar columna id basada en el nombre del archivo si no existe
    if 'id' not in df_simple.columns:
        df_simple['id'] = df_simple['path'].apply(os.path.basename)
    if 'id' not in df_complex.columns:
        df_complex['id'] = df_complex['path'].apply(os.path.basename)

    # Renombrar columnas de etiquetas para evitar conflictos
    df_simple = df_simple.rename(columns={'label': 'label-simple'})
    df_complex = df_complex.rename(columns={'label': 'label-complex'})
    
    # Unir etiquetas simples y complejas usando el id
    df_labels_merged = pd.merge(df_simple, df_complex[['id', 'label-complex']], on='id', how='inner')
    
    # Unir con el texto limpio para crear el dataset maestro
    df_master = pd.merge(df_labels_merged, df_text[['id', 'text_clean']], on='id', how='inner')
    
    # Crear directorio de salida si no existe
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Guardar el dataset completo para entrenamiento final
    complete_data_path = os.path.join(OUTPUT_DIR, "complete-data.csv")
    df_master.to_csv(complete_data_path, index=False)
    print(f"Dataset completo guardado en: {complete_data_path}")

    # Separar conjunto de prueba manteniendo balance de clases complejas
    df_temp, df_test = train_test_split(
        df_master, 
        test_size=0.15, 
        random_state=42, 
        stratify=df_master['label-complex'] 
    )
    
    # Separar entrenamiento y validacion del conjunto restante
    df_train, df_val = train_test_split(
        df_temp, 
        test_size=0.176, 
        random_state=42, 
        stratify=df_temp['label-complex']
    )
    
    # Guardar los archivos de particion
    df_train.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    df_val.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    df_test.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)
    
    print(f"Splits guardados en: {OUTPUT_DIR}")

if __name__ == "__main__":
    create_master_splits()