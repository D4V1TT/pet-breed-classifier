Pet Breed Classifier

Fine-grained classification of 37 cat and dog breeds from a single photo, built end to end: data exploration, transfer learning with an ImageNet-pretrained ResNet18, controlled experiments, error analysis, ONNX export and quantisation, and a Gradio web demo with an API.

Dataset: Oxford-IIIT Pet (7,349 images, 37 breeds).

Test accuracy: 89.7% (frozen-backbone baseline: 87.9%)
Macro F1: 0.896
Deployed model: ONNX FP32, 44.8 MB, CPU inference
INT8 model: 11.3 MB (4× smaller), no accuracy loss

Demo

Full write-up: [Technical report](reports/technical_report.md)

Quick start: run the demo

1. Clone and install
With uv (recommended):
git clone https://github.com/D4V1TT/pet-breed-classifier.git
cd pet-breed-classifier
uv sync

Or with pip:

python -m venv .venv
pip install -r requirements.txt
pip install -e .

2. Download the trained model

Model files are not stored in git because of their size. Download them from the [Releases page](https://github.com/D4V1TT/pet-breed-classifier/releases/tag/v1.0) and place them like this:

models/
|-- model.onnx      - download from the release
|-- classes.json    -  already in the repo

Release files:
File                         Description
model.onnx               Deployed model (ONNX FP32), used by the app
model_int8.onnx          Dynamically quantised INT8 version (smaller, slower on CPU)
resnet18_ft_lr3e-5.pt    PyTorch state_dict of the same model

3. Run the app

uv run python app.py

Open http://127.0.0.1:7860, upload a cat or dog photo, and get the top-3 breeds with probabilities.

4. Use it as an API

Gradio exposes the app as an HTTP API automatically (see "Use via API" at the bottom of the page). From Python, with the app running:

from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:7860/")

result = client.predict(handle_file("my_dog.jpg"), api_name="/predict")

print(result)   # predicted breed and top-3 probabilities

Reproduce training

The dataset (~800 MB) is downloaded automatically by torchvision into [data](data/) the first time a notebook runs.

Run the notebooks in order, from the [notebooks/](notebooks/) folder (they use paths relative to it):

uv run jupyter lab

Notebook                                                   What it does
data_exploration.ipynb          Image sizes, class balance, stratified split check, augmentation preview 
training.ipynb                  Sanity check, frozen-backbone training, three fine-tuning runs (backbone LR 1e-4 / 3e-5 / 1e-5), test evaluation
evaluation.ipynb                Per-class metrics, confusion matrix, inspection of misclassified images
optimization.ipynb              ONNX export and verification, INT8 quantisation, size/latency/accuracy comparison

Checkpoints are saved to [parameters/](parameters/) (git-ignored). Training was done on an NVIDIA RTX 4060 Laptop GPU (8 GB); each fine-tuning run takes ~10 minutes. A CPU works but is much slower.

Reproducibility note: the data split is fixed (seed=0), but training seeds were not fixed, so re-running training gives slightly different numbers (in my re-run, the selected model scored 89.9% instead of 89.7%). The released model is from the re-run. See the technical report for details.

Results

Experiments

All fine-tuning runs start from the trained frozen-backbone model; head LR 1e-3, Adam, 10 epochs. Model selection was done on the validation set; the test set was used once at the end.

Experiment                                  Best val acc           Lowest val loss       Test acc
Frozen backbone (head only)                   0.902                    0.302              0.879
Fine-tune, backbone LR 1e-4                   0.913                    0.296              0.881
Fine-tune, backbone LR 3e-5 (selected)        0.920                    0.244              0.897
Fine-tune, backbone LR 1e-5                   0.920                    0.255              0.898

Backbone LR 1e-4 overfits (validation loss rises while training accuracy reaches 99%); lowering it to 3e-5 or 1e-5 keeps validation loss stable. Differences under ~1 point are within noise (test standard error ~ 0.5 points).

[Learning curves](reports/learning_curves.png)

Error analysis

Errors concentrate in visually similar breed groups: American Pit Bull Terrier / Staffordshire Bull Terrier / American Bulldog, and Birman / Ragdoll / Siamese.

[Confusion matrix](reports/confusion_matrix.png)

Optimisation (before vs after)

Version                  Size (MB)     CPU latency (ms/image)   Test acc
PyTorch FP32              44.9              61.9                 0.899
ONNX FP32                 44.8              79.0                 0.898
ONNX INT8 (dynamic)       11.3              433                  0.899

INT8 is 4x smaller with no accuracy loss, but dynamic quantisation is slower on CPU for convolutional networks; static quantisation is the planned next step. The app deploys ONNX FP32, which needs only onnxruntime for inference.

Project structure


pet-breed-classifier/
|-- src/pet_breed_classifier/
|   |-- data.py          # transforms, stratified train/val split, dataloaders
|   |-- model.py         # ResNet18 setup, training loop, evaluation, predictions
|   |-- visualize.py     # image display and misclassification plots
|   |-- benchmark.py     # latency benchmark and ONNX accuracy helpers
|-- notebooks/           # exploration, training, evaluation, optimisation
|-- reports/
|   |-- technical_report.md
|   |-- *.png            # figures
|   |-- *.json           # training histories per experiment
|-- models/
|   |-- classes.json     # class names (model.onnx from the release goes here)
|-- app.py               # Gradio demo + API
|-- pyproject.toml / uv.lock / requirements.txt
|-- README.md

Main dependencies

Pytorch, torchvision, scikit-learn, NumPy, pandas, matplotlib, ONNX, ONNX Runtime, Gradio, Jupyter. Exact versions are in [uv.lock](uv.lock) / [requirements.txt](requirements.txt).

Limitations:

-The model always predicts one of the 37 breeds, even for images without a cat or dog.

-Single training run per setting; small differences between settings are within run-to-run noise.

-Dynamic INT8 quantisation reduces size but not CPU latency.F