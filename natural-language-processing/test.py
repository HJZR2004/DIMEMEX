import os
import sys

# configuracion de las rutas

# configuracion de la ruta actual del script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Obtener raiz del proyecto (subir un nivel desde natural-language-processing)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../"))
# Definir ruta de src donde esta nlp_utils.py
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# Agregar src al path de Python para poder importar modulos de ahi
sys.path.append(SRC_DIR)

#imports necesarios
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix

from nlp_utils import BetoClassifier, MemeDataset


VERSION = int(input("Ingrese la version del modelo a testear (ej: 1, 2, 3, 4): "))

# Definicion de directorios
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "nlp")
# La carpeta del modelo especifico: DIMEMEX/models/v1/
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", f"v{VERSION}")
TEST_FILE = "test.csv"

# Hiperparametros
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
MAX_LEN = 128
BATCH_SIZE = 16

# Funcion para ejecutar predicciones sin calcular gradientes
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

# Pipeline que configura y ejecuta la evaluacion segun la tarea
def run_test_pipeline(task_name, device):
    print("\n" + "="*50)
    print(f" EVALUANDO MODELO: {task_name.upper()} (Versión {VERSION})")
    print("="*50)

    # Configuracion de rutas y columnas segun la tarea
    # Estructura esperada: models/v1/beto_simple_v1.pth
    if task_name == "simple":
        filename = f"beto_simple_v{VERSION}.pth"
        target_col = "label-simple"
        classes_names = ["None", "Inappropriate", "Hate"]
    elif task_name == "complex":
        filename = f"beto_complex_v{VERSION}.pth"
        target_col = "label-complex"
        classes_names = ["None", "Inapp", "Sexism", "Racism", "Classicism", "Other"]
    else:
        print(f"Tarea desconocida: {task_name}")
        return

    model_path = os.path.join(MODEL_DIR, filename)

    # Validar existencia del archivo del modelo
    if not os.path.exists(model_path):
        print(f"Error: No encuentro el archivo del modelo en {model_path}")
        return

    # Validar existencia del archivo de datos
    test_path = os.path.join(DATA_DIR, TEST_FILE)
    if not os.path.exists(test_path):
        print(f"Error: No existe el archivo de datos en {test_path}")
        return
        
    df_test = pd.read_csv(test_path)
    print(f"Datos cargados: {len(df_test)} muestras")
    
    n_classes = len(classes_names)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Instanciar dataset y dataloader
    # Pasamos label_col para que sepa que columna leer (simple o complex)
    test_dataset = MemeDataset(df_test, tokenizer, MAX_LEN, label_col=target_col)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    print(f"Cargando modelo desde: {model_path}")
    
    # Inicializar modelo y cargar pesos
    model = BetoClassifier(n_classes, MODEL_NAME)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error cargando pesos del modelo: {e}")
        return

    model = model.to(device)
    
    # Obtener predicciones
    y_true, y_pred = test_model(model, test_loader, device)
    
    # Imprimir metricas de evaluacion
    print("\n--- Reporte de Clasificacion ---")
    print(classification_report(y_true, y_pred, target_names=classes_names))
    
    print("\n--- Matriz de Confusion ---")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

if __name__ == "__main__":
    # Configuracion de dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de prueba: {device}")
    
    tasks_to_test = ["simple", "complex"]
    
    for task in tasks_to_test:
        try:
            run_test_pipeline(task, device)
        except Exception as e:
            print(f"Ocurrio un error evaluando {task}: {e}")