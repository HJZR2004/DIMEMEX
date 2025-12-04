import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
import time
import sys
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# Configuracion de rutas dinamicas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../"))

SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# Agregar el directorio actual al path para importar nlp_utils
sys.path.append(SRC_DIR)
from nlp_utils import BetoClassifier, MemeDataset

# Definicion de directorios de datos y modelos
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "nlp")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "v3")
COMPLETE_DATA_FILE = "complete-data.csv"

# Hiperparametros globales
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5

# Funcion de entrenamiento por epoca
def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["label"].to(device)
        
        outputs = model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        
    return correct_predictions.double() / n_examples, sum(losses) / len(losses)

# Funcion de evaluacion del modelo
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["label"].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            
    return correct_predictions.double() / n_examples, sum(losses) / len(losses)

# Preparacion de datasets segun modo produccion o experimental
def prepare_datasets(task_name, production_mode):
    if task_name == "simple":
        target_col = "label-simple"
    elif task_name == "complex":
        target_col = "label-complex"
    else:
        raise ValueError("Tarea desconocida")

    if production_mode:
        # Carga dataset completo para produccion
        full_path = os.path.join(DATA_DIR, COMPLETE_DATA_FILE)
        if not os.path.exists(full_path):
            print(f"Error: No existe {full_path}")
            return None, None, None
            
        df_full = pd.read_csv(full_path)
        
        # Usar casi todo para train dejando un minimo para validacion tecnica
        df_train, df_val = train_test_split(
            df_full, 
            test_size=0.05, 
            random_state=42, 
            stratify=df_full[target_col]
        )
    else:
        # Carga splits de experimentacion estandar
        train_path = os.path.join(DATA_DIR, "train.csv")
        val_path = os.path.join(DATA_DIR, "val.csv")
        
        if not os.path.exists(train_path) or not os.path.exists(val_path):
            print("Error: No existen train.csv o val.csv")
            return None, None, None
            
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)

    return df_train, df_val, target_col

# Ejecucion del pipeline de entrenamiento completo
def run_pipeline(task_name, production_mode, device):
    mode_str = "PROD" if production_mode else "EXP"
    print(f"\nIniciando tarea {task_name.upper()} en modo {mode_str}")

    # Configurar nombre del modelo
    suffix = "_prod" if production_mode else "_exp"
    model_filename = f"beto_{task_name}{suffix}.pth"
    model_save_path = os.path.join(MODEL_DIR, model_filename)

    # Evitar re-entrenamiento si ya existe
    if os.path.exists(model_save_path):
        print(f"Aviso: El modelo {model_filename} ya existe. Saltando...")
        return

    # Preparar datos
    df_train, df_val, target_col = prepare_datasets(task_name, production_mode)
    if df_train is None:
        return

    print(f"Train size: {len(df_train)} | Val size: {len(df_val)}")
    
    # Calcular pesos de clase para balanceo
    all_labels = df_train[target_col].values
    classes = np.unique(all_labels)
    n_classes = len(classes)
    
    print(f"Calculando pesos para {n_classes} clases...")
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=all_labels)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    
    # Preparar tokenizador y dataloaders
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader = DataLoader(MemeDataset(df_train, tokenizer, MAX_LEN, target_col), 
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MemeDataset(df_val, tokenizer, MAX_LEN, target_col), 
                            batch_size=BATCH_SIZE)

    # Inicializar modelo y optimizador
    model = BetoClassifier(n_classes, MODEL_NAME).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(device)

    # Bucle de entrenamiento
    best_acc = 0
    os.makedirs(MODEL_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        start = time.time()
        
        train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, len(df_train))
        val_acc, val_loss = eval_model(model, val_loader, loss_fn, device, len(df_val))
        
        print(f"Epoca {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Guardar siempre el mejor modelo basado en accuracy de validacion
        if val_acc > best_acc:
            torch.save(model.state_dict(), model_save_path)
            best_acc = val_acc
            print(f"Nuevo mejor modelo guardado en {model_save_path}")

    print(f"Finalizado {model_filename}. Mejor Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    # Configuracion de dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo global: {device}")

    # Lista de configuraciones a ejecutar
    scenarios = [
        {"task": "simple", "prod": False},
        {"task": "complex", "prod": False},
        {"task": "simple", "prod": True},
        {"task": "complex", "prod": True}
    ]

    for sc in scenarios:
        try:
            run_pipeline(sc["task"], sc["prod"], device)
        except Exception as e:
            print(f"Error critico en escenario {sc}: {e}")
            
    print("\nTodos los procesos de entrenamiento han finalizado.")