from copy import deepcopy

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

from config import (
    TRAIN_DIR,
    TEST_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    PERSISTENT_WORKERS,
    PREFETCH_FACTOR,
    VALIDATION_SPLIT,
    RANDOM_SEED,
)

from utils.transforms import train_transform, validation_transform, test_transform


def get_dataloaders():
    """
    Returns:
        train_loader
        val_loader
        test_loader
        class_names
    """

    # -------------------------
    # Full Training Dataset
    # -------------------------
    full_dataset = ImageFolder(
        root=TRAIN_DIR,
        transform=train_transform
    )

    class_names = full_dataset.classes

    # -------------------------
    # Train / Validation Split
    # -------------------------
    total_size = len(full_dataset)
    val_size = int(total_size * VALIDATION_SPLIT)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )

    # Validation should NOT use augmentation
    val_dataset.dataset = deepcopy(full_dataset)
    val_dataset.dataset.transform = validation_transform

    # -------------------------
    # Test Dataset
    # -------------------------
    test_dataset = ImageFolder(
        root=TEST_DIR,
        transform=test_transform
    )

    # -------------------------
    # DataLoaders
    # -------------------------
    common_args = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
    }

    if NUM_WORKERS > 0:
        common_args["persistent_workers"] = PERSISTENT_WORKERS
        common_args["prefetch_factor"] = PREFETCH_FACTOR

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **common_args
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_args
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_args
    )

    # -------------------------
    # Dataset Info
    # -------------------------
    print("\nDataset Summary")
    print("-" * 40)
    print(f"Classes        : {class_names}")
    print(f"Training Images: {len(train_dataset)}")
    print(f"Validation     : {len(val_dataset)}")
    print(f"Testing        : {len(test_dataset)}")
    print("-" * 40)

    return train_loader, val_loader, test_loader, class_names