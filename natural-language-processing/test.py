import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix
import os

from nlp_utils import BetoClassifier, MemeDataset

# Configuracion de rutas
DATA_DIR = "../data/processed/splits"
TEST_FILE = "test.csv"
MODEL_DIR = "../models"

# Hiperparametros
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
MAX_LEN = 128
BATCH_SIZE = 16

# Funcion para ejecutar predicciones
def test_model(model, data_loader, device):
    model = model.eval()
    
    predictions = []
    real_values = []
    
    print("Ejecutando inferencia en el conjunto de test...")
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["label"].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            real_values.extend(targets.cpu().tolist())
            
    return real_values, predictions

# Pipeline de test
def run_test_pipeline(task_name, device):
    print("\n" + "="*50)
    print(f" EVALUANDO MODELO: {task_name.upper()}")
    print("="*50)

    if task_name == "simple":
        model_path = os.path.join(MODEL_DIR, "beto_simple.pth")
        target_col = "label-simple"
        classes_names = ["None", "Inappropriate", "Hate"]
    elif task_name == "complex":
        model_path = os.path.join(MODEL_DIR, "beto_complex.pth")
        target_col = "label-complex"
        classes_names = ["None", "Inapp", "Sexism", "Racism", "Classicism", "Other"]
    else:
        print(f"Tarea desconocida: {task_name}")
        return

    if not os.path.exists(model_path):
        print(f"Error: No encuentro el archivo del modelo en {model_path}")
        return

    test_path = os.path.join(DATA_DIR, TEST_FILE)
    if not os.path.exists(test_path):
        print(f"Error: No existe el archivo de datos en {test_path}")
        return
        
    df_test = pd.read_csv(test_path)
    print(f"Datos cargados: {len(df_test)} muestras")
    
    n_classes = len(classes_names)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Instanciamos dataset importado
    test_dataset = MemeDataset(df_test, tokenizer, MAX_LEN, label_col=target_col)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    print(f"Cargando modelo desde: {model_path}")
    # Instanciamos modelo importado
    model = BetoClassifier(n_classes, MODEL_NAME)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    y_true, y_pred = test_model(model, test_loader, device)
    
    print("\n--- Reporte de Clasificacion ---")
    print(classification_report(y_true, y_pred, target_names=classes_names))
    
    print("\n--- Matriz de Confusion ---")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de prueba: {device}")
    
    tasks_to_test = ["simple", "complex"]
    
    for task in tasks_to_test:
        try:
            run_test_pipeline(task, device)
        except Exception as e:
            print(f"Ocurrio un error evaluando {task}: {e}")