import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.model import LSTMModel
from models.utils import save_model
from src.preprocessing import clean_text  # If needed

def train_model(train_data, train_labels, vocab_size, embedding_dim, hidden_dim, output_dim, epochs=5):
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # Convert data to DataLoader if it's not
    train_loader = DataLoader(list(zip(train_data, train_labels)), batch_size=32, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for data, labels in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    save_model(model, "models/saved/lstm_model.pth")

