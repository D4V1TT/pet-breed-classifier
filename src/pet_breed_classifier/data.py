import torch
import numpy as np
from torchvision.models import ResNet18_Weights
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from sklearn.model_selection import train_test_split


def get_transforms(train: bool):
    """return the training(augmented) or evaluation transform."""
    if train:
        train_transform = v2.Compose([
            v2.RandomResizedCrop(224, scale=(0.6, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(ResNet18_Weights.DEFAULT.transforms().mean,
                         ResNet18_Weights.DEFAULT.transforms().std)
        ])

        return train_transform

    return ResNet18_Weights.DEFAULT.transforms()


def get_dataloaders(root: Path,
                    batch_size: int = 32,
                    val_fraction: float = 0.2,
                    seed: int = 0,
                    num_workers: int = 0):
    """return train, validation, test dataloaders, plus a list of class names"""
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    dataset_train = datasets.OxfordIIITPet(root=root,
                                           split="trainval",
                                           transform=train_transform,
                                           download=True)
    dataset_val = datasets.OxfordIIITPet(root=root,
                                         split="trainval",
                                         transform=val_transform,
                                         download=True)
    testset = datasets.OxfordIIITPet(root=root,
                                     split="test",
                                     transform=val_transform,
                                     download=True)

    labels = np.asarray(dataset_train._labels)
    indices = np.arange(len(labels))

    train_indices, val_indices = train_test_split(indices,
                                                  test_size=val_fraction,
                                                  random_state=seed,
                                                  stratify=labels)

    trainset = Subset(dataset_train, train_indices)
    valset = Subset(dataset_val, val_indices)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(valset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    return train_loader, val_loader, test_loader, dataset_train.classes