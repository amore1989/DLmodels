from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, test_data, test_labels):
    model.eval()
    with torch.no_grad():
        predictions = model(test_data)
        _, predicted = torch.max(predictions, 1)
        
        accuracy = accuracy_score(test_labels, predicted)
        precision = precision_score(test_labels, predicted, average='weighted')
        recall = recall_score(test_labels, predicted, average='weighted')
        f1 = f1_score(test_labels, predicted, average='weighted')

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
