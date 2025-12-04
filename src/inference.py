import torch
import easyocr
import os
import sys
from transformers import AutoTokenizer

# Configurar path para importar módulos locales (nlp_utils y utils)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from nlp_utils import BetoClassifier
from utils import clean_text, preprocess_image_for_ocr

MODEL_VERSION = "v4" 
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Construcción de la ruta: .../DIMEMEX/models/v4/
# Subimos dos niveles desde src/inference.py para llegar a la raiz, luego models, luego v4
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../"))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", MODEL_VERSION)

class MemePredictor:
    def __init__(self):
        print(f"Inicializando motor en: {DEVICE}")
        print(f"Buscando modelos versión {MODEL_VERSION} en: {MODEL_DIR}")
        
        # Cargar OCR (Singleton)
        self.reader = easyocr.Reader(['es', 'en'], gpu=(DEVICE.type == 'cuda'))
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Cache para no recargar modelos .pth
        self.loaded_models = {}
        
        # Mapas de etiquetas
        self.labels_map = {
            "simple": ["None", "Inappropriate", "Hate"],
            "complex": ["None", "Inapp", "Sexism", "Racism", "Classicism", "Hate"]
        }

    def _get_model_instance(self, task):
        # Si ya está en RAM, devolverlo
        if task in self.loaded_models:
            return self.loaded_models[task]
        
        # Construir nombre del archivo: beto_simple_v4.pth
        filename = f"beto_{task}_{MODEL_VERSION}.pth"
        path = os.path.join(MODEL_DIR, filename)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"No se encontró el modelo para la tarea '{task}' en {path}")

        print(f"Cargando modelo desde: {path}")
        
        n_classes = len(self.labels_map[task])
        
        # Instanciar arquitectura
        model = BetoClassifier(n_classes, MODEL_NAME)
        
        # Cargar pesos
        # map_location es vital para evitar errores si entrenaste en GPU y corres en CPU
        try:
            model.load_state_dict(torch.load(path, map_location=DEVICE))
        except Exception as e:
            raise RuntimeError(f"Error al leer el archivo .pth: {e}")
            
        model.to(DEVICE)
        model.eval()
        
        self.loaded_models[task] = model
        return model

    def predict(self, image_file, task="simple"):
        # 1. Resetear puntero del archivo
        image_file.seek(0)
        
        # 2. Pre-procesamiento de Imagen (Visión)
        proc_img, raw_img = preprocess_image_for_ocr(image_file)
        
        if proc_img is None:
            return {"error": "Error procesando la imagen (archivo corrupto o formato inválido)"}

        # 3. OCR con Fallback
        try:
            # Intento 1: Imagen procesada
            ocr_result = self.reader.readtext(proc_img, detail=0, paragraph=True)
            raw_text = " ".join(ocr_result)
            
            # Si leyó muy poco (<3 chars), intentar con la original
            if len(raw_text) < 3:
                ocr_result = self.reader.readtext(raw_img, detail=0, paragraph=True)
                raw_text = " ".join(ocr_result)
        except Exception as e:
            return {"error": f"Fallo en OCR: {e}"}

        # 4. Limpieza de Texto (NLP)
        text_ready = clean_text(raw_text)
        
        # 5. Tokenización
        encoding = self.tokenizer.encode_plus(
            text_ready,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)
        
        # 6. Inferencia del Modelo
        try:
            model = self._get_model_instance(task)
            
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, idx = torch.max(probs, dim=1)
                
            label_str = self.labels_map[task][idx.item()]
            
            return {
                "ocr_text": raw_text,
                "clean_text": text_ready,
                "label": label_str,
                "confidence": conf.item(),
                "probabilities": probs.cpu().numpy()[0],
                "all_labels": self.labels_map[task]
            }
        except Exception as e:
            return {"error": f"Error en inferencia del modelo: {e}"}