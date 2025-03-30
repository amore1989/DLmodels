# Starter script: preprocessing.py
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

def preprocess_for_train(df):
    """
    Preprocess the training dataset.
    """
    # Example preprocessing: clean text
    df['text'] = df['text'].apply(clean_text)
    # Add more preprocessing steps as needed
    return df

def preprocess_for_test(df):
    """
    Preprocess the test dataset.
    """
    # Example preprocessing: clean text
    df['text'] = df['text'].apply(clean_text)
    # Add more preprocessing steps if needed
    return df

def preprocess_for_validation(df):
    """
    Preprocess the validation dataset.
    """
    # Example preprocessing: clean text
    df['text'] = df['text'].apply(clean_text)
    # Add more preprocessing steps if needed
    return df

def preprocess_for_inference(df):
    """
    Preprocess the inference dataset.
    """
    # Example preprocessing: clean text
    df['text'] = df['text'].apply(clean_text)
    # Add more preprocessing steps if needed
    return df

def preprocessing_for_evaluation(df):
    """
    Preprocess the evaluation dataset.
    """
    # Example preprocessing: clean text
    df['text'] = df['text'].apply(clean_text)
    # Add more preprocessing steps if needed
    return df



# EXAMPLE OF USING THIS SCRIPT
import pandas as pd
from src.preprocessing import (
    preprocess_for_train,
    preprocess_for_test,
    preprocess_for_validation,
    preprocess_for_inference,
    preprocessing_for_evaluation
)

# Example dataset paths
TRAIN_DATA_PATH = "data/raw/train.csv"
TEST_DATA_PATH = "data/raw/test.csv"
VALIDATION_DATA_PATH = "data/raw/validation.csv"
INFERENCE_DATA_PATH = "data/raw/inference.csv"
EVALUATION_DATA_PATH = "data/raw/evaluation.csv"

# Load datasets
train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)
validation_df = pd.read_csv(VALIDATION_DATA_PATH)
inference_df = pd.read_csv(INFERENCE_DATA_PATH)
evaluation_df = pd.read_csv(EVALUATION_DATA_PATH)

# Preprocess datasets
train_df = preprocess_for_train(train_df)
test_df = preprocess_for_test(test_df)
validation_df = preprocess_for_validation(validation_df)
inference_df = preprocess_for_inference(inference_df)
evaluation_df = preprocessing_for_evaluation(evaluation_df)

# Save processed datasets
train_df.to_csv("data/processed/train_cleaned.csv", index=False)
test_df.to_csv("data/processed/test_cleaned.csv", index=False)
validation_df.to_csv("data/processed/validation_cleaned.csv", index=False)
inference_df.to_csv("data/processed/inference_cleaned.csv", index=False)
evaluation_df.to_csv("data/processed/evaluation_cleaned.csv", index=False)

print("Preprocessing completed. Cleaned datasets saved.")

