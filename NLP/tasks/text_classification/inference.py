import torch
from model import SimpleTextClassifier
from preprocessing import clean_text

def predict(text, model, tokenizer):
    model.eval()
    text = clean_text(text)
    tokenized_text = tokenizer(text)
    input_tensor = torch.tensor(tokenized_text).unsqueeze(0)  # Adding batch dimension
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    return predicted_class

if __name__ == "__main__":
    model = SimpleTextClassifier(vocab_size=5000, embed_size=128, hidden_size=128, num_classes=2)
    # Load a pretrained model or trained model weights
    model.load_state_dict(torch.load("path/to/saved_model.pth"))
    text = "Your input text here"
    prediction = predict(text, model, tokenizer=lambda x: x.split())  # Replace tokenizer
    print(f"Predicted class: {prediction}")
