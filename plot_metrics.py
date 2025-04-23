import matplotlib.pyplot as plt
from config import EPOCHS

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot training and validation metrics."""
    print("\nPlotting training and validation metrics...")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss')
    plt.plot(range(1, EPOCHS + 1), val_losses, label='Val Loss')
    plt.title('Loss over Epochs (with Regularization)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, EPOCHS + 1), val_accuracies, label='Train Accuracy')
    plt.title('Accuracy over Epochs (with Regularization)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('metrics_plot.png')
    print("Metrics plot saved as metrics_plot.png")