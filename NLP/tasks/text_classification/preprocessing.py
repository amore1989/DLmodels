import re
import string
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Text cleaning function
def clean_text(text):
    """
    Preprocesses text by lowercasing, removing punctuation, and filtering stopwords.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return " ".join(tokens)  # Reconstruct text

# Preprocessing function
def preprocess_text_classification_data(file_path, output_dir, test_size=0.2, val_size=0.1):
    """
    Preprocesses a text classification dataset:
    - Cleans text
    - Encodes labels
    - Splits into train, validation, and test sets
    - Saves preprocessed files

    :param file_path: Path to the input dataset (CSV file)
    :param output_dir: Directory to save processed datasets
    :param test_size: Fraction of data for testing
    :param val_size: Fraction of train data for validation
    """
    # Load dataset
    df = pd.read_csv(file_path)
    
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns")

    # Apply text cleaning
    df['text'] = df['text'].apply(clean_text)

    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])

    # Split dataset
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['label'])
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=42, stratify=train_df['label'])

    # Save processed datasets
    train_df.to_csv(f"{output_dir}/train_cleaned.csv", index=False)
    val_df.to_csv(f"{output_dir}/val_cleaned.csv", index=False)
    test_df.to_csv(f"{output_dir}/test_cleaned.csv", index=False)

    print(f"Preprocessing complete. Data saved in {output_dir}")

    return train_df, val_df, test_df, label_encoder


# Example Usage: run_preprocessing.py

from src.preprocessing import preprocess_text_classification_data

# File paths
RAW_DATA_PATH = "data/raw/text_classification_dataset.csv"
OUTPUT_DIR = "data/processed"

# Run preprocessing
train_df, val_df, test_df, label_encoder = preprocess_text_classification_data(RAW_DATA_PATH, OUTPUT_DIR)

# Display dataset information
print(f"Train Set: {len(train_df)} samples")
print(f"Validation Set: {len(val_df)} samples")
print(f"Test Set: {len(test_df)} samples")
print(f"Classes: {label_encoder.classes_}")
