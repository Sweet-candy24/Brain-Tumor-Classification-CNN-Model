import torch

from models.cnn import BrainCNN
from utils.dataset import get_dataloaders
from utils.metrics import calculate_metrics, print_report

from config import DEVICE, BEST_MODEL


@torch.no_grad()
def main():

    _, _, test_loader, class_names = get_dataloaders()

    model = BrainCNN().to(DEVICE)

    model.load_state_dict(torch.load(BEST_MODEL, map_location=DEVICE))

    model.eval()

    y_true = []
    y_pred = []

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())

    metrics = calculate_metrics(y_true, y_pred)

    print("\nTest Results")
    print("-" * 40)

    print(f"Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall   : {metrics['recall']*100:.2f}%")
    print(f"F1 Score : {metrics['f1']*100:.2f}%")

    cm = print_report(y_true, y_pred, class_names)

    print("\nConfusion Matrix")
    print(cm)


if __name__ == "__main__":
    main()