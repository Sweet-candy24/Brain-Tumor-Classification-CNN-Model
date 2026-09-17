import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import DEVICE, BEST_MODEL
from models.cnn import BrainCNN


CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])


def load_model():
    model = BrainCNN(num_classes=4)

    model.load_state_dict(
        torch.load(BEST_MODEL, map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()

    return model


@torch.no_grad()
def predict(image_path):

    model = load_model()

    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    outputs = model(input_tensor)

    probabilities = F.softmax(outputs, dim=1)

    confidence, prediction = torch.max(probabilities, dim=1)

    predicted_class = CLASS_NAMES[prediction.item()]
    confidence = confidence.item() * 100

    print("\nPrediction")
    print("-" * 35)
    print(f"Class      : {predicted_class}")
    print(f"Confidence : {confidence:.2f}%")

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"{predicted_class} ({confidence:.2f}%)")
    plt.show()


def main():

    parser = argparse.ArgumentParser(
        description="Brain Tumor Classification"
    )

    parser.add_argument(
        "image",
        type=str,
        help="Path to MRI image",
    )

    args = parser.parse_args()

    predict(args.image)


if __name__ == "__main__":
    main()