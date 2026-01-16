import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def plot_training_history(history, save_path=None):
    if history is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["loss"], label="Training Loss", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    axes[0].set_title("Model Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(history.history["accuracy"], label="Training Accuracy", linewidth=2)
    axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
    axes[1].set_title("Model Accuracy", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, prediction_threshold=0.5, save_path=None):
    y_pred_binary = (y_pred > prediction_threshold).astype(int).flatten()
    y_true_binary = y_true.astype(int).flatten()
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(f"Confusion Matrix - {class_names[0]} vs {class_names[1]}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_prediction_distribution(predictions, prediction_threshold=0.5, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(predictions, bins=30, alpha=0.7, color="blue", edgecolor="black")
    ax.axvline(
        x=prediction_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Decision Boundary ({prediction_threshold})",
    )
    ax.set_xlabel("Prediction Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Model Predictions", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_pred, save_path=None):
    y_true_binary = y_true.astype(int).flatten()
    y_pred_flat = y_pred.flatten()
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred_flat)
    roc_auc = roc_auc_score(y_true_binary, y_pred_flat)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return roc_auc


def generate_report(y_true, y_pred, class_names, prediction_threshold=0.5, use_roc_auc=True):
    y_pred_binary = (y_pred > prediction_threshold).astype(int).flatten()
    y_true_binary = y_true.astype(int).flatten()
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary, average="weighted", zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, average="weighted", zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, average="weighted", zero_division=0)
    if use_roc_auc:
        y_pred_flat = y_pred.flatten()
        roc_auc = roc_auc_score(y_true_binary, y_pred_flat)
    classification_report(y_true_binary, y_pred_binary, target_names=class_names, digits=4, zero_division=0)
