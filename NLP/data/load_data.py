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
