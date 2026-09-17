import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
)

from sklearn.preprocessing import label_binarize

from config import DEVICE, BEST_MODEL
from models.cnn import BrainCNN
from utils.dataset import get_dataloaders

os.makedirs("outputs/plots", exist_ok=True)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]


@torch.no_grad()
def main():

    _, _, test_loader, _ = get_dataloaders()

    model = BrainCNN().to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL, map_location=DEVICE))
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probs = F.softmax(outputs, dim=1)

        preds = probs.argmax(dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    print(classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES
    ))

    # -------------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------------

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig("outputs/plots/confusion_matrix.png")

    plt.close()

    # -------------------------------------------------------
    # ROC
    # -------------------------------------------------------

    y_bin = label_binarize(
        y_true,
        classes=[0,1,2,3]
    )

    plt.figure(figsize=(8,6))

    for i in range(4):

        fpr, tpr, _ = roc_curve(
            y_bin[:,i],
            y_prob[:,i]
        )

        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{CLASS_NAMES[i]} (AUC={roc_auc:.3f})"
        )

    plt.plot([0,1],[0,1],"k--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curves")

    plt.legend()

    plt.tight_layout()

    plt.savefig("outputs/plots/roc_curve.png")

    plt.close()

    # -------------------------------------------------------
    # Precision Recall
    # -------------------------------------------------------

    plt.figure(figsize=(8,6))

    for i in range(4):

        precision, recall, _ = precision_recall_curve(
            y_bin[:,i],
            y_prob[:,i]
        )

        plt.plot(
            recall,
            precision,
            label=CLASS_NAMES[i]
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title("Precision-Recall Curves")

    plt.legend()

    plt.tight_layout()

    plt.savefig("outputs/plots/pr_curve.png")

    plt.close()

    # -------------------------------------------------------
    # Per-class Accuracy
    # -------------------------------------------------------

    class_acc = cm.diagonal() / cm.sum(axis=1)

    plt.figure(figsize=(7,5))

    plt.bar(CLASS_NAMES, class_acc * 100)

    plt.ylabel("Accuracy (%)")

    plt.title("Per-Class Accuracy")

    plt.ylim(0,100)

    plt.tight_layout()

    plt.savefig("outputs/plots/class_accuracy.png")

    plt.close()

    print("\nGraphs saved to outputs/plots/")


if __name__ == "__main__":
    main()