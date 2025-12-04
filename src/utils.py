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

def preprocess_image_for_ocr(file_obj):

    try:
        # Leer el archivo (puede ser UploadedFile de Streamlit o bytes)
        if hasattr(file_obj, 'read'):
            file_bytes = np.asarray(bytearray(file_obj.read()), dtype=np.uint8)
        else:
            file_bytes = np.asarray(bytearray(file_obj), dtype=np.uint8)
        
        img = cv2.imdecode(file_bytes, 1)
        
        if img is None:
            return None, None

        # 1. Upscaling (Mantenemos esto, es vital)
        img_resized = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # 2. Convertir a escala de grises
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # aumentamos el contraste para mejorar la detección de texto 
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_img = clahe.apply(gray)
        
        # denoinsing con fastNlMeansDenoising para conservar mejor detalles finos
        denoised = cv2.fastNlMeansDenoising(contrast_img, None, h=5, templateWindowSize=7, searchWindowSize=21)
        
        # retornamos la imagen denoised y la original redimensionada en escala de grises
        return denoised, img

    except Exception as e:
        print(f"Error pre-procesamiento: {e}")
        return None, None