# Starter script: generate_embeddings.py
import numpy as np
import gensim.downloader as api

def generate_word_embeddings(text_data, model_name="glove-wiki-gigaword-50"):
    """Generates word embeddings using a pre-trained model from Gensim."""
    model = api.load(model_name)
    embeddings = [model[word] for word in text_data if word in model]
    return np.array(embeddings)

if __name__ == "__main__":
    sample_text = ["hello", "world", "this", "is", "a", "test"]
    embeddings = generate_word_embeddings(sample_text)
    print("Sample Embeddings Shape:", embeddings.shape)
