# Starter script: load_data.py
import pandas as pd
import os

def load_data(file_path):
    """Loads a dataset from a CSV or TXT file."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.readlines()
    else:
        raise ValueError("Unsupported file format")

if __name__ == "__main__":
    sample_data_path = "../data/raw/sample.csv"
    if os.path.exists(sample_data_path):
        data = load_data(sample_data_path)
        print("Sample Data Loaded:")
        print(data.head() if isinstance(data, pd.DataFrame) else data[:5])

import pandas

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

# Example usage
from load_data import load_train_data, load_test_data

train_df = load_train_data('data/train_data1.csv', 'data/train_data2.csv')
test_df = load_test_data('data/test_data1.csv', 'data/test_data2.csv')

# Preprocessing (if needed)
train_df = preprocess_for_train(train_df)
test_df = preprocess_for_test(test_df)

# Create PyTorch datasets and dataloaders
train_dataset = TextDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer, max_len)
test_dataset = TextDataset(test_df['text'].tolist(), test_df['label'].tolist(), tokenizer, max_len)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)





