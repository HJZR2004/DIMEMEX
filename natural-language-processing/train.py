import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import os
import time

# Configuracion de rutas
SPLITS_DIR = "../data/processed/splits"
MODEL_DIR = "../models"

# Hiperparametros globales
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5

# Clase para cargar el dataset
class MemeDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, label_col):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_col = label_col # Columna dinamica segun la tarea
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # Cargar texto limpio
        text = str(self.df.loc[index, 'text_clean'])
        # Cargar label dinamico (simple o complex)
        label = int(self.df.loc[index, self.label_col])
        
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

# Modelo basado en BETO
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

# Funciones de entrenamiento y evaluacion
def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["label"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        
    return correct_predictions.double() / n_examples, sum(losses) / len(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["label"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            
    return correct_predictions.double() / n_examples, sum(losses) / len(losses)

def run_training_pipeline(task_name, device):
    print(f"\nIniciando entrenamiento para tarea: {task_name}")

    # Configurar columna objetivo y ruta de modelo segun tarea
    if task_name == "simple":
        target_col = "label-simple"
        model_save_path = os.path.join(MODEL_DIR, "beto_simple.pth")
    elif task_name == "complex":
        target_col = "label-complex"
        model_save_path = os.path.join(MODEL_DIR, "beto_complex.pth")
    else:
        print(f"Tarea desconocida: {task_name}")
        return

    # Verificar si el modelo ya existe
    if os.path.exists(model_save_path):
        print(f"El modelo ya existe en: {model_save_path}. Saltando...")
        return

    # Cargar los splits ya existentes (train.csv y val.csv)
    train_path = os.path.join(SPLITS_DIR, "train.csv")
    val_path = os.path.join(SPLITS_DIR, "val.csv")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("Error: No se encuentran los archivos train.csv o val.csv en splits.")
        return

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    # Detectar numero de clases basado en la columna especifica
    n_classes = df_train[target_col].nunique()
    print(f"Clases detectadas ({target_col}): {n_classes}")
    print(f"Datos de entrenamiento: {len(df_train)} | Validacion: {len(df_val)}")

    # Preparar tokenizador y loaders
    # Se pasa target_col al dataset para que sepa que etiqueta leer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader = DataLoader(MemeDataset(df_train, tokenizer, MAX_LEN, target_col), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MemeDataset(df_val, tokenizer, MAX_LEN, target_col), batch_size=BATCH_SIZE)

    # Inicializar modelo
    model = BetoClassifier(n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss().to(device)

    # Bucle de entrenamiento
    best_acc = 0
    os.makedirs(MODEL_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        start = time.time()
        
        train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, len(df_train))
        val_acc, val_loss = eval_model(model, val_loader, loss_fn, device, len(df_val))
        
        print(f"Train Acc: {train_acc:.4f} | Loss: {train_loss:.4f}")
        print(f"Val   Acc: {val_acc:.4f} | Loss: {val_loss:.4f}")
        
        if val_acc > best_acc:
            torch.save(model.state_dict(), model_save_path)
            best_acc = val_acc
            print(f"Modelo guardado: {model_save_path}")

    print(f"Finalizado {task_name}. Mejor Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    # Configuracion de dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # Lista de tareas a ejecutar secuencialmente
    tasks_to_run = ["simple", "complex"]

    for task in tasks_to_run:
        try:
            run_training_pipeline(task, device)
        except Exception as e:
            print(f"Error critico en tarea {task}: {e}")
            continue
            
    print("\nEntrenamiento completado para todas las tareas.")