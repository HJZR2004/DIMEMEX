# src/nlp_utils.py
import torch.nn as nn
from torch.utils.data import Dataset
import torch
from transformers import AutoModel

class BetoClassifier(nn.Module):
    def __init__(self, n_classes, model_name="dccuchile/bert-base-spanish-wwm-cased"):
        super(BetoClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.out(self.drop(output.pooler_output))

class MemeDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, label_col):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_col = label_col # Columna dinamica segun la tarea
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        text = str(self.df.loc[index, 'text_clean'])
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