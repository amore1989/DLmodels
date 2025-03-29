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


