import pandas as pd
import json
import os
import re

# Configuracion de rutas dinamicas basadas en la ubicacion de este archivo
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../"))

# Definicion de rutas de entrada y salida
INPUT_JSON_PATH = os.path.join(PROJECT_ROOT, "data", "train", "train_data.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, "nlp","cleaned-ocr.csv")

def clean_text(text):
    # Funcion para limpiar texto extraido de las imagenes
    if pd.isna(text) or str(text).strip() == "":
        return "sin texto"
    
    text = str(text)

    # Eliminar URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Eliminar menciones de usuarios
    text = re.sub(r'@\w+', '', text)
    
    # Normalizacion de risas
    text = re.sub(r'(ja|je|ha|he){2,}', 'jaja', text)
    
    # Reduccion de caracteres repetidos
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Eliminar caracteres basura de OCR
    text = re.sub(r'[|_~*^>\[\]]', ' ', text)
    
    # Reemplazar saltos de linea
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Reemplazar espacios multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    if text == "":
        return "sin texto"
        
    return text

def build_text():
    # Validar existencia del archivo de entrada
    if not os.path.exists(INPUT_JSON_PATH):
        print(f"Error: No se encuentra el archivo {INPUT_JSON_PATH}")
        return

    # Cargar archivo JSON
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convertir a DataFrame
    df = pd.DataFrame(data)

    # Renombrar columna de ID para consistencia
    df.rename(columns={'MEME-ID': 'id'}, inplace=True)
    
    # Aplicar limpieza a texto y descripciones
    df['text_clean'] = df['text'].apply(clean_text)
    df['desc_clean'] = df['description'].apply(clean_text)
    
    # Seleccion final de columnas
    final_df = df[['id', 'text', 'text_clean', 'description', 'desc_clean']]
    
    # Crear directorio si no existe y guardar CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Archivo guardado en: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    build_text()