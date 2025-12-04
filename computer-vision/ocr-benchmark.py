import pandas as pd
import easyocr
import os
import Levenshtein
from tqdm import tqdm
import cv2
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# configuracion de rutas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../"))

# Rutas de entrada y salida
INPUT_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed","datasets","dataset-simple.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "computer-vision")
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, "ocr_benchmark_results.csv")


# definimos batch size y num workers(hilos)
BATCH_SIZE = 8
NUM_WORKERS = 4

reader = easyocr.Reader(['es', 'en'], gpu=True)


# funciones para preprocesar imagenes y calcular similitud
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None: return None, None

    # upscaling (x2), grayscale, denoise, binarization
    # nos ayuda a mejorar OCR en imagenes pequeñas o con ruido
    try:
        img_resized = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return binary, img
    except Exception:
        return None, None

# funcion para normalizar texto (minusculas, eliminar caracteres especiales)
def normalize_text(text):
    if pd.isna(text) or text == "": return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9áéíóúñü ]', '', text)
    return text.strip()

# funcion para calcular similitud entre textos
def calculate_similarity(text1, text2):
    t1 = normalize_text(text1)
    t2 = normalize_text(text2)
    if not t1 and not t2: return 1.0
    return Levenshtein.ratio(t1, t2)


# 
class OCRDataset(Dataset):
    def __init__(self, csv_path, project_root):
        self.df = pd.read_csv(csv_path)
        self.project_root = project_root
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row['path']
        real_text = row['text']
        
        # Construir ruta
        clean_rel_path = rel_path.replace("../", "").lstrip("/")
        full_path = os.path.join(self.project_root, clean_rel_path)
        image_id = os.path.basename(rel_path)
        
        # Pre-procesar AQUI (en el worker)
        if os.path.exists(full_path):
            proc_img, raw_img = preprocess_image(full_path)
            valid = (proc_img is not None)
        else:
            proc_img, raw_img, valid = None, None, False
            
        return {
            "id": image_id,
            "path": rel_path,
            "real_text": real_text,
            "proc_img": proc_img,
            "raw_img": raw_img,
            "valid": valid
        }

def custom_collate(batch):
    # Como las imagenes tienen tamaños diferentes, no podemos hacer un Tensor stack.
    # Retornamos una lista simple.
    return batch

# ==========================================
# 3. EJECUCIÓN
# ==========================================
def run_benchmark():
    print(f"Iniciando Benchmark con Workers: {NUM_WORKERS}")
    
    if not os.path.exists(INPUT_CSV_PATH):
        print("Error: No encuentro CSV de entrada")
        return

    dataset = OCRDataset(INPUT_CSV_PATH, PROJECT_ROOT)
    
    # DataLoader: Aqui ocurre la magia de la velocidad
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        collate_fn=custom_collate # Importante para imagenes de distinto tamano
    )

    results = []

    print(f"Procesando {len(dataset)} imagenes en lotes...")
    
    for batch in tqdm(loader):
        # 'batch' es una lista de diccionarios gracias a custom_collate
        for item in batch:
            extracted_text = ""
            similarity = 0.0
            
            if item["valid"]:
                try:
                    # Intento 1: Imagen procesada
                    res = reader.readtext(item["proc_img"], detail=0, paragraph=True)
                    text = " ".join(res)
                    
                    if len(text) > 3:
                        extracted_text = text
                    else:
                        # Intento 2: Imagen original
                        res_raw = reader.readtext(item["raw_img"], detail=0, paragraph=True)
                        extracted_text = " ".join(res_raw)
                    
                    similarity = calculate_similarity(item["real_text"], extracted_text)
                    
                except Exception:
                    extracted_text = "ERROR_OCR"
            else:
                extracted_text = "IMG_ERROR"

            results.append({
                'id': item["id"],
                'path': item["path"],
                'real_text': item["real_text"],
                'extracted_text': extracted_text,
                'similarity_score': round(similarity, 4)
            })

    # Guardar
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_CSV_PATH, index=False)
    
    avg = df_results['similarity_score'].mean()
    print(f"Finalizado. Similitud Promedio: {avg*100:.2f}%")
    print(f"Guardado en: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    run_benchmark()