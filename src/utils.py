import re
import cv2
import numpy as np
import pandas as pd

def clean_text(text):
    """Limpieza estándar para BERT."""
    if not text or pd.isna(text): return "sin texto"
    text = str(text).lower()
    # Eliminar URLs y usuarios
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    # Normalizar risas
    text = re.sub(r'(ja|je|ha|he|lo){2,}', 'jaja', text)
    # Eliminar basura de OCR
    text = re.sub(r'[|_~*^>\[\]]', ' ', text)
    # Espacios y saltos
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else "sin texto"

def preprocess_image_for_ocr(file_bytes):
    """
    Recibe bytes (desde Streamlit) y aplica filtros de OpenCV.
    Retorna: (imagen_binarizada, imagen_original_cv2)
    """
    # Convertir bytes a array numpy para OpenCV
    file_bytes = np.asarray(bytearray(file_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1) # 1 = Color BGR
    
    if img is None: return None, None

    # Pipeline de Mejora (Igual al benchmark)
    try:
        # 1. Upscaling (2x)
        img_resized = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # 2. Escala de Grises
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # 3. Denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # 4. Binarización Adaptativa
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return binary, img
    except Exception as e:
        print(f"Error en pre-procesamiento: {e}")
        return None, img