import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report, confusion_matrix
import os
import seaborn as sns
import matplotlib.pyplot as plt

# cambiar aquí para elegir la tarea (simple/complex)
TASK_NAME = "simple"

# Rutas
DATA_DIR = "../data/processed/splits"
TEST_FILE = "test.csv"

# Configuración automática
if TASK_NAME == "simple":
    MODEL_PATH = "../models/beto_simple.pth"
    CLASSES_NAMES = ["None", "Inappropriate", "Hate"]
elif TASK_NAME == "complex":
    MODEL_PATH = "../models/beto_complex.pth"
    CLASSES_NAMES = ["None", "Inapp", "Sexism", "Racism", "Classicism", "Other"]

# Hiperparámetros (Deben ser idénticos al entrenamiento)
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
MAX_LEN = 128
BATCH_SIZE = 16


# define dataset y modelo
class MemeDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        text = str(self.df.loc[index, 'text_clean'])
        label = int(self.df.loc[index, 'label'])
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BetoClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BetoClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.pooler_output
        return self.out(self.drop(pooled_output))

# funcion para testear el modelo
def test_model(model, data_loader, device):
    model = model.eval()
    
    predictions = []
    real_values = []
    
    print("Ejecutando predicciones...")
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["label"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            
            # Guardamos predicciones y valores reales en CPU para scikit-learn
            predictions.extend(preds.cpu().tolist())
            real_values.extend(targets.cpu().tolist())
            
    return real_values, predictions

# bloque principal
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testando en: {device}")
    
    # 1. Cargar Datos de Test
    test_path = os.path.join(DATA_DIR, TEST_FILE)
    if not os.path.exists(test_path):
        print("Error: No existe test.csv en splits.")
        exit()
        
    df_test = pd.read_csv(test_path)
    print(f"Datos de Test cargados: {len(df_test)} muestras")
    
    # 2. Preparar Modelo
    n_classes = len(CLASSES_NAMES)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_loader = DataLoader(MemeDataset(df_test, tokenizer, MAX_LEN), batch_size=BATCH_SIZE)
    
    # 3. Cargar Pesos Guardados
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No encuentro el modelo entrenado en {MODEL_PATH}")
        print("   Ejecuta train_modular_fixed.py primero.")
        exit()
        
    print(f"Cargando pesos desde: {MODEL_PATH}")
    model = BetoClassifier(n_classes)
    # map_location es vital por si entrenaste en GPU y testeas en CPU
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    
    # 4. Ejecutar Test
    y_true, y_pred = test_model(model, test_loader, device)
    
    # 5. Reporte de Resultados
    print("\n" + "="*30)
    print(f" RESULTADOS FINALES: TAREA {TASK_NAME.upper()}")
    print("="*30)
    
    print(classification_report(y_true, y_pred, target_names=CLASSES_NAMES))
    
    # Opcional: Matriz de Confusión simple en texto
    print("\nMatriz de Confusión (Filas=Real, Columnas=Predicción):")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)