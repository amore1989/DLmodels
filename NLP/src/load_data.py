# src/load_data.py
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.optim as optim

def load_train_data(file_path1, file_path2):
    """
    Load two training datasets and combine them.
    """
    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)
    # Combine both datasets
    combined_train_df = pd.concat([df1, df2], ignore_index=True)
    return combined_train_df

def load_test_data(file_path1, file_path2):
    """
    Load two test datasets and combine them.
    """
    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)
    # Combine both datasets
    combined_test_df = pd.concat([df1, df2], ignore_index=True)
    return combined_test_df

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokenized_text = self.tokenizer(
            text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        return tokenized_text['input_ids'].squeeze(), label

def tokenizer(text):
    """
    Tokenize the input text.
    """
    # Example tokenizer: split by whitespace
    return {"input_ids": torch.tensor([ord(c) for c in text])}

def train_model():
    # Load and preprocess data
    df = load_train_data('data/train_data1.csv', 'data/train_data2.csv')
    tokenizer_func = tokenizer  # Replace with HuggingFace tokenizer if needed
    max_len = 100

    X_train, X_val, y_train, y_val = train_test_split(df['text'], df['label'], test_size=0.2)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)

    # Convert labels to tensors
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # Create datasets
    train_dataset = TextDataset(X_train.tolist(), y_train.tolist(), tokenizer_func, max_len)
    val_dataset = TextDataset(X_val.tolist(), y_val.tolist(), tokenizer_func, max_len)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Model setup
    model = SimpleTextClassifier(vocab_size=5000, embed_size=128, hidden_size=128, num_classes=len(label_encoder.classes_))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(10):
        model.train()
        for batch_idx, (texts, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/10], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

class SimpleTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super(SimpleTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out
