from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, labels, class_names, title="Confusion Matrix", show=True, save=True, save_path=None):
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))

    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.xticks(rotation=45, ha='right')  # <-- This ensures x-axis labels don’t overlap
    plt.tight_layout()                   # <-- Adjusts plot to fit labels
    plt.title(title)
    if show:
        plt.show()
    if save:
        plt.savefig(save_path or "confusion_matrix.png", dpi=600, bbox_inches='tight')
    print(f"Saved confusion matrix as {save_path or 'confusion_matrix.png'}")