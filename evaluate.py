import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def evaluate_model(model, val_loader, device):
    """Evaluate the model on the validation set."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        eval_loop = tqdm(val_loader, desc="Evaluating")
        for input_ids, labels in eval_loop:
            input_ids = input_ids.to(device)
            logits = model(input_ids)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    final_val_acc = accuracy_score(all_labels, all_preds)
    print(f"\nFinal Validation Accuracy: {final_val_acc:.4f}")
    return final_val_acc

def save_model(model, path="models/cnn_spam_model_regularized.pt"):
    """Save the model's state dictionary."""
    try:
        torch.save(model.state_dict(), path)
        print(f"Model state dictionary saved successfully to {path}")
    except Exception as e:
        print(f"Error saving model: {e}")