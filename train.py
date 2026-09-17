import os
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    RANDOM_SEED,
    BEST_MODEL,
    LAST_MODEL,
    PATIENCE,
    USE_AMP,
)

from models.cnn import BrainCNN
from utils.dataset import get_dataloaders
from utils.train_utils import seed_everything, EarlyStopping


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, leave=False)

    for images, labels in pbar:

        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=USE_AMP):

            outputs = model(images)

            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        correct += predicted.eq(labels).sum().item()

        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader), 100 * correct / total


@torch.no_grad()
def validate(model, loader, criterion):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    for images, labels in loader:

        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=USE_AMP):

            outputs = model(images)

            loss = criterion(outputs, labels)

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        correct += predicted.eq(labels).sum().item()

        total += labels.size(0)

    return running_loss / len(loader), 100 * correct / total


def main():

    seed_everything(RANDOM_SEED)

    os.makedirs(BEST_MODEL.parent, exist_ok=True)

    writer = SummaryWriter("runs/BrainCNN")

    train_loader, val_loader, _, _ = get_dataloaders()

    model = BrainCNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    early_stop = EarlyStopping(PATIENCE)

    best_acc = 0.0

    history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
    "lr": [],
    }

    for epoch in range(NUM_EPOCHS):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
        )

        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.2f}% | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.2f}%"
        )

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)

        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)

        if val_acc > best_acc:

            best_acc = val_acc

            torch.save(model.state_dict(), BEST_MODEL)

            print("Best model saved.")

        torch.save(model.state_dict(), LAST_MODEL)

        if early_stop(val_loss):

            print("Early stopping.")

            break

    writer.close()

    os.makedirs("outputs", exist_ok=True)

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        "outputs/training_history.csv",
        index=False,
    )

    print("\nTraining history saved.")


if __name__ == "__main__":
    main()