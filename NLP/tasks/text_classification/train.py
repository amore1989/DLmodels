import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from model import SimpleTextClassifier
from preprocessing import load_and_preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn

    
    

def train_model():
    # Load and preprocess data
    df = load_and_preprocess_data('path/to/your/dataset.csv')
    tokenizer = lambda text: text.split()  # You can replace this with HuggingFace tokenizer
    max_len = 100
    X_train, X_val, y_train, y_val = train_test_split(df['text'], df['label'], test_size=0.2)
    
    train_data = TextDataset(X_train.tolist(), y_train.tolist(), tokenizer, max_len)
    val_data = TextDataset(X_val.tolist(), y_val.tolist(), tokenizer, max_len)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    # Encode labels
    LabelEncoder = LabelEncoder()
    y_train = LabelEncoder.fit_transform(y_train)
    y_val = LabelEncoder.transform(y_val)
    # Convert labels to tensors
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    # Create datasets
    train_dataset = TextDataset(X_train.tolist(), y_train.tolist(), tokenizer, max_len)
    val_dataset = TextDataset(X_val.tolist(), y_val.tolist(), tokenizer, max_len)

                


    # Model setup
    model = SimpleTextClassifier(vocab_size=5000, embed_size=128, hidden_size=128, num_classes=len(set(df['label'])))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(10):
        model.train()
        for batch_idx, (texts, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/10], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for texts, labels in val_loader:
                outputs = model(texts)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {correct/total:.4f}")

if __name__ == "__main__":
    train_model()

