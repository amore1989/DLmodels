# src/load_data.py
import pandas as pd

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
