import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import DEVICE, BEST_MODEL, NUM_CLASSES, CLASS_NAMES
from models.cnn import BrainCNN


# =============================================================================
# Image Preprocessing
# =============================================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])


# =============================================================================
# Load Trained Model
# =============================================================================

def load_model():
    if not BEST_MODEL.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at: {BEST_MODEL}"
        )

    model = BrainCNN(num_classes=NUM_CLASSES)

    model.load_state_dict(
        torch.load(
            BEST_MODEL,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    return model


# =============================================================================
# Prediction
# =============================================================================

@torch.no_grad()
def predict(image_path):
    model = load_model()

    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

    outputs = model(input_tensor)

    probabilities = F.softmax(outputs, dim=1)

    confidence, prediction = torch.max(probabilities, dim=1)

    predicted_class = CLASS_NAMES[prediction.item()]
    confidence_percentage = confidence.item() * 100

    print("\nPrediction")
    print("-" * 35)
    print(f"Class      : {predicted_class}")
    print(f"Confidence : {confidence_percentage:.2f}%")
    print(f"Device     : {DEVICE}")

    # Display the input image with prediction
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(
        f"{predicted_class} "
        f"({confidence_percentage:.2f}%)"
    )
    plt.show()


# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Brain Tumor MRI Classification using a trained CNN"
    )

    parser.add_argument(
        "image",
        type=str,
        help="Path to the MRI image"
    )

    args = parser.parse_args()

    predict(args.image)


if __name__ == "__main__":
    main()
