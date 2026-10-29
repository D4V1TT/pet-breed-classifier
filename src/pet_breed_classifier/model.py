import torch
import torch.optim as optim
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from torch import nn
from typing import Tuple

def get_model(num_classes: int, device: str = 'cpu', *, freeze: bool = False) -> nn.Module:
    """returns resnet18 model for fine-tuning"""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = model.to(device)

    return model

def get_criterion():
    """returns criterion for fine-tuning"""
    criterion = nn.CrossEntropyLoss()
    return criterion

def get_optimizer(parameters, lr: float = 0.001):
    """returns optimizer for fine-tuning"""
    optimizer = optim.Adam(parameters, lr=lr)
    return optimizer


def train_one_epoch(model: nn.Module,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    data_loader: torch.utils.data.DataLoader,
                    device: torch.device) -> Tuple[float, float]:

    model.train()
    total_loss, correct, total_n = 0.0, 0, 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_n += labels.size(0)

        correct += (logits.argmax(dim=1) == labels).sum().item()

    return total_loss / total_n, correct / total_n


def evaluate(model: nn.Module,
             criterion: nn.Module,
             data_loader: torch.utils.data.DataLoader,
             device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total_n = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)


            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total_n += labels.size(0)

            correct += (logits.argmax(dim=1) == labels).sum().item()

    return total_loss / total_n, correct / total_n

def get_predictions(model: nn.Module,
                    data_loader: torch.utils.data.DataLoader,
                    device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)

            logits = model(images)

            predictions = logits.argmax(dim=1).cpu().numpy()
            labels = labels.numpy()

            all_labels.append(labels)
            all_predictions.append(predictions)

    return np.concatenate(all_labels), np.concatenate(all_predictions)
