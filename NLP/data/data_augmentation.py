from textblob import TextBlob
import random

def augment_text(text, n=1):
    """
    Simple paraphrasing using TextBlob.
    """
    augmented_texts = []
    blob = TextBlob(text)

    for _ in range(n):
        # Slightly alter the text using TextBlob's n-gram functionality
        augmented_text = blob.translate(to='en')  # You can use different translations or methods here
        augmented_texts.append(str(augmented_text))

    return augmented_texts

# Example usage
original_text = "The quick brown fox jumps over the lazy dog."
augmented_texts = augment_text(original_text, n=3)
print(augmented_texts)
