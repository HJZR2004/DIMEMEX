import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import os
import time

# Opciones: "simple" | "complex"
TASK_NAME = "simple" 

DATA_DIR = "../data/processed/splits"
TRAIN_FILE = "train.csv"
VAL_FILE = "val.csv"

if TASK_NAME == "simple":
    MODEL_SAVE_PATH = "../models/beto_simple.pth"
    CLASSES_NAMES = ["None", "Inappropriate", "Hate"]
elif TASK_NAME == "complex":
    MODEL_SAVE_PATH = "../models/beto_complex.pth"
    CLASSES_NAMES = ["None", "Inapp", "Sexism", "Racism", "Classicism", "Other"]
else:
    raise ValueError("TASK_NAME debe ser 'simple' o 'complex'")

# Hiperparámetros
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5

# clase para cargar el dataset
class MemeDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # Cargamos directo del DF ya procesado
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


# modelo basado en BETO que añade una capa de dropout y una capa lineal final
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


# funciones de entrenamiento y evaluación
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


if __name__ == "__main__":

    # Verificar si el modelo ya existe
    if os.path.exists(MODEL_SAVE_PATH):
        print("\n" + "="*50)
        print(f"El modelo ya existe en: {MODEL_SAVE_PATH}")
        print("="*50 + "\n")
        exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando en: {device}")
    
    # Cargar datos pre-procesados
    print(f"Cargando splits desde {DATA_DIR}...")
    train_path = os.path.join(DATA_DIR, TRAIN_FILE)
    val_path = os.path.join(DATA_DIR, VAL_FILE)
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("ERROR: No encuentro train.csv o val.csv en la carpeta splits.")
        print("Ejecuta primero el script de partición.")
        exit()

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    
    print(f"   Train: {len(df_train)} | Val: {len(df_val)}")

    # Verificar Clases
    n_classes = df_train['label'].nunique()
    print(f"Clases detectadas en CSV: {n_classes}")
    
    # Validación simple: Si configuraste 'simple' (3 clases) pero el CSV tiene 6, alerta.
    if TASK_NAME == "simple" and n_classes > 3:
        print("ADVERTENCIA: Estás entrenando modo SIMPLE pero el CSV parece tener muchas clases.")
    
    # Preparar Loaders
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader = DataLoader(MemeDataset(df_train, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MemeDataset(df_val, tokenizer, MAX_LEN), batch_size=BATCH_SIZE)
    
    # Modelo
    model = BetoClassifier(n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    # Entrenamiento
    best_acc = 0
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        start = time.time()
        
        train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, len(df_train))
        val_acc, val_loss = eval_model(model, val_loader, loss_fn, device, len(df_val))
        
        print(f"  Train Acc: {train_acc:.4f} | Loss: {train_loss:.4f}")
        print(f"  Val   Acc: {val_acc:.4f} | Loss: {val_loss:.4f}")
        
        if val_acc > best_acc:
            os.makedirs("../modelos", exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            best_acc = val_acc
            print(f"Modelo guardado: {MODEL_SAVE_PATH}")
            
    print("\nFin.")