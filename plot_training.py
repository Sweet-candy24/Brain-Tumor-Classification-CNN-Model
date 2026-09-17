import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs/plots", exist_ok=True)

history = pd.read_csv("outputs/training_history.csv")

# ---------------- Loss ----------------

plt.figure(figsize=(8,5))

plt.plot(history["epoch"], history["train_loss"], linewidth=2, label="Train")
plt.plot(history["epoch"], history["val_loss"], linewidth=2, label="Validation")

plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("outputs/plots/loss_curve.png")
plt.close()

# ---------------- Accuracy ----------------

plt.figure(figsize=(8,5))

plt.plot(history["epoch"], history["train_acc"], linewidth=2, label="Train")
plt.plot(history["epoch"], history["val_acc"], linewidth=2, label="Validation")

plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("outputs/plots/accuracy_curve.png")
plt.close()

# ---------------- Learning Rate ----------------

plt.figure(figsize=(8,5))

plt.plot(history["epoch"], history["lr"], linewidth=2)

plt.title("Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.grid(True)

plt.tight_layout()
plt.savefig("outputs/plots/learning_rate.png")
plt.close()

print("Training plots saved to outputs/plots/")