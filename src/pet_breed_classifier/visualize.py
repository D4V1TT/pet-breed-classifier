import torch
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torchvision.models import ResNet18_Weights
from torch.utils.data import Subset

def imshow(img: torch.Tensor, title: str = None):
    image = img.cpu()

    image = image.numpy().transpose((1, 2, 0))
    mean = ResNet18_Weights.DEFAULT.transforms().mean
    std = ResNet18_Weights.DEFAULT.transforms().std

    image = image * std + mean
    image = np.clip(image, 0, 1)

    plt.imshow(image)

    if title is not None:
        plt.title(title)



def show_errors(raw_test: torchvision.datasets.OxfordIIITPet,
                y_true: np.ndarray,
                y_pred: np.ndarray,
                true_name: str,
                pred_name: str,
                n: int = 4) -> None:

    true_idx = raw_test.class_to_idx[true_name]
    pred_idx = raw_test.class_to_idx[pred_name]

    errors = np.where((y_true == true_idx) & (y_pred == pred_idx))[0][0:n]

    fig, ax = plt.subplots(1, len(errors), figsize=(10 *n , 3 ))

    for ax_i, (img, label) in zip(ax, Subset(raw_test, errors)):
        ax_i.imshow(img)
        ax_i.set_title(f"true: {true_name}; pred: {pred_name}")
        ax_i.axis('off')

    plt.show()


