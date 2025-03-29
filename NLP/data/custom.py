import torch
from torch.utils.data import Dataset
from src.preprocessing import clean_text
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=256):
        self.data = pd.read_csv(file_path)  # Assuming CSV with 'text' and 'label' columns
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']
        
        # Clean the text
        cleaned_text = clean_text(text)

        # Tokenize the text
        tokens = self.tokenizer.encode_plus(
            cleaned_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = tokens['input_ids'].squeeze()  # Remove unnecessary dimensions
        attention_mask = tokens['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Usage example
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = CustomDataset('data/processed/your_dataset.csv', tokenizer)
