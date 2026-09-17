from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }


def print_report(y_true, y_pred, class_names):
    print("\nClassification Report")
    print("-" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))

    return confusion_matrix(y_true, y_pred)