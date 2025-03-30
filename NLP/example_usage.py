# using load.data.py example

import torch
from src.load_data import load_train_data, load_test_data, TextDataset, tokenizer, train_model

# File paths (replace with actual paths)
train_file_1 = "data/train_data1.csv"
train_file_2 = "data/train_data2.csv"
test_file_1 = "data/test_data1.csv"
test_file_2 = "data/test_data2.csv"

# Load training and test data
train_df = load_train_data(train_file_1, train_file_2)
test_df = load_test_data(test_file_1, test_file_2)

print("Training Data Loaded:")
print(train_df.head())

print("Test Data Loaded:")
print(test_df.head())

# Tokenizer function and max length
tokenizer_func = tokenizer
max_len = 100

# Prepare dataset & dataloader
train_dataset = TextDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer_func, max_len)
test_dataset = TextDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer_func, max_len)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Example: Fetch a batch from DataLoader
for batch_idx, (texts, labels) in enumerate(train_loader):
    print(f"Batch {batch_idx+1}:")
    print("Tokenized Input:", texts)
    print("Labels:", labels)
    break  # Show only one batch for demonstration

# Train the model
train_model()






