import time
import torch
from typing import Callable


def benchmark(predict_fn: Callable,
              x: torch.Tensor,
              warmup: int = 10,
              runs: int = 100) -> float:
    """Average milliseconds per call of predict_fn(x)"""
    for _ in range(warmup):
        predict_fn(x)

    start = time.perf_counter()
    for _ in range(runs):
        predict_fn(x)

    return (time.perf_counter() - start) / runs * 1000

def onnx_accuracy(session, loader) -> float:
    correct, total = 0, 0
    for images, labels in loader:
        logits = session.run(None, {"image":images.numpy()})[0]
        correct += (logits.argmax(axis=1) == labels.numpy()).sum()
        total += logits.shape[0]

    return  correct / total