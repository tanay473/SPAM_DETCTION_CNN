import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import LR, EPOCHS, WEIGHT_DECAY
from model import CNNClassifier

def train_model(model, train_loader, val_loader, device):
    """Train and validate the CNN model."""
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    print("\nStarting training with regularization...")
    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        total_train_correct = 0
        total_train_samples = 0

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for input_ids, labels in train_loop:
            input_ids, labels = input_ids.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total_train_correct += (preds == labels).sum().item()
            total_train_samples += labels.size(0)

            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_accuracy = total_train_correct / total_train_samples
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        total_val_correct = 0
        total_val_samples = 0

        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]")
            for input_ids, labels in val_loop:
                input_ids, labels = input_ids.to(device), labels.to(device)

                logits = model(input_ids)
                loss = criterion(logits, labels)

                total_val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                total_val_correct += (preds == labels).sum().item()
                total_val_samples += labels.size(0)

                val_loop.set_postfix(loss=loss.item())

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_accuracy = total_val_correct / total_val_samples
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)

        print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss = {avg_train_loss:.4f}, Train Acc = {avg_train_accuracy:.4f}, Val Loss = {avg_val_loss:.4f}, Val Acc = {avg_val_accuracy:.4f}")

    print("\nTraining finished.")
    return train_losses, val_losses, train_accuracies, val_accuracies
