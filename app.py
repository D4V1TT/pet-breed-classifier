import json
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort

from pet_breed_classifier.data import get_transforms

MODELS_DIR = Path(__file__).parent / "models"

session = ort.InferenceSession(str(MODELS_DIR / 'model.onnx'), providers=['CPUExecutionProvider'])
classes = json.loads((MODELS_DIR / 'classes.json').read_text())
transform = get_transforms(train=False)

def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()

def predict(image) -> dict[str, float]:
    x = transform(image).unsqueeze(0).numpy()
    logits = session.run(None, {"image": x})[0][0]
    probs = softmax(logits)
    top3 = probs.argsort()[::-1][:3]
    return {classes[i]: float(probs[i]) for i in top3}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a cat or dog photo"),
    outputs=gr.Label(num_top_classes=3, label="Predicted breed"),
    flagging_mode = "never",
    title="Pet Breed Classifier",
    description="Fine-tuned ResNet18 (ONNX) on the Oxford-III Pet dataset, 37 breeds, 89.7% test accuracy."
)

if __name__ == "__main__":
    demo.launch()