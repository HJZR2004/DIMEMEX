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
    """
    Versión optimizada para memes con subtítulos oscuros/complejos.
    """
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
        
        # 3. CAMBIO CLAVE: Aumentar Contraste (CLAHE) en lugar de Binarizar agresivamente
        # CLAHE (Contrast Limited Adaptive Histogram Equalization) mejora el texto
        # sin destruir los bordes como lo hace el Threshold puro.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_img = clahe.apply(gray)
        
        # 4. Denoising suave (bajamos h de 10 a 5 para no borrar letras finas)
        denoised = cv2.fastNlMeansDenoising(contrast_img, None, h=5, templateWindowSize=7, searchWindowSize=21)
        
        # Retornamos la imagen contrastada (gris) en lugar de binarizada (blanco/negro)
        # EasyOCR a veces prefiere grises con buen contraste que binarización forzada.
        return denoised, img

    except Exception as e:
        print(f"Error pre-procesamiento: {e}")
        return None, None