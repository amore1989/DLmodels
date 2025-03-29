import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

def clean_text(text):
    """
    Clean the input text by lowering the case, removing punctuation, and stopwords.
    """
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

def load_and_preprocess_data(file_path):
    """
    Load dataset from file and preprocess it.
    Assuming CSV with columns: 'text' and 'label'
    """
    df = pd.read_csv(file_path)
    df['text'] = df['text'].apply(clean_text)
    return df

# Example usage
if __name__ == "__main__":
    df = load_and_preprocess_data('/workspaces/DLmodels/NLP/data/raw/Sheet_1.csv')
    print(df.head())

