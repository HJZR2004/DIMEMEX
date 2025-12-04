import pandas as pd
import json
import os
import re
"""
este codigo nos ayuda a limpiar el texto extraido de las imagenes,
y guardarlo en un archivo csv llamado cleaned-ocr.csv dentro de la carpeta data/processed
"""



BASE_DIR = "../data"
TRAIN_FOLDER = "train" 
OUTPUT_DIR = "../data/processed/nlp"



def clean_text(text):

    """
    funcion que nos ayuda a limpiar el texto que se extrae de las imagenes
    """
    if pd.isna(text) or str(text).strip() == "":
        return "sin texto"
    
    text = str(text)


    # Eliminar URLs (ruido de marcas de agua)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Eliminar menciones de usuarios 
    text = re.sub(r'@\w+', '', text)
    
    # Normalización de Risas (Crucial en México) a "jaja"
    text = re.sub(r'(ja|je|ha|he){2,}', 'jaja', text)
    
    # Reducción de caracteres repetidos "Hooooola" -> "hoola" 
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Eliminar caracteres OCR basura y puntuación irrelevante
    
    text = re.sub(r'[|_~*^>\[\]]', ' ', text)
    
    # Saltos de línea a espacio
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Espacios múltiples a uno solo
    text = re.sub(r'\s+', ' ', text).strip()
    
    if text == "":
        return "sin texto"
        
    return text



def build_text():
    
    # Cargamos el json
    json_path = os.path.join(BASE_DIR, TRAIN_FOLDER, f"{TRAIN_FOLDER}_data.json")
    

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convertir a DataFrame
    df = pd.DataFrame(data)

    # Renombramos 'MEME-ID' a 'id' para consistencia
    df.rename(columns={'MEME-ID': 'id'}, inplace=True)
    
    # Limpiamos el texto y las descripciones
    df['text_clean'] = df['text'].apply(clean_text)
    df['desc_clean'] = df['description'].apply(clean_text)
    

    # Selección final de columnas
    final_df = df[['id', 'text', 'text_clean', 'description', 'desc_clean']]
    
    # Guardar
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "cleaned-ocr.csv")
    final_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    build_text()