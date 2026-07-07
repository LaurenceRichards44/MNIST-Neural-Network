# lernet

A from-scratch, fully connected neural network library built in NumPy. Designed for learning and experimentation — no PyTorch or TensorFlow required.

## Features

- **Fully connected layers** with configurable sizes and per-layer activations
- **Activations:** linear, ReLU, sigmoid, tanh, softmax
- **Loss functions:** cross-entropy (classification), MSE (regression)
- **Optimizers:** SGD, Momentum, Adam
- **L2 regularization** via a `lambda_regularization` parameter
- **Mini-batch gradient descent** training with progress bars
- **Model save/load** to compressed `.npz` files
- **Training history** tracking across multiple runs
- **Built-in plotting:** loss/accuracy curves, confusion matrices, model summary

## Installation

From the `lernet` directory (where `pyproject.toml` lives):

```bash
pip install -e .
```

You'll also need these dependencies:

```bash
pip install numpy matplotlib scikit-learn tqdm torchvision
```

## Quick start — MNIST

```python
import numpy as np
import torchvision
from lernet import Network

# Load and preprocess MNIST
train = torchvision.datasets.MNIST(root="data/", train=True, download=True)
test = torchvision.datasets.MNIST(root="data/", train=False, download=True)

X_train = train.data.numpy().astype(np.float32).reshape(len(train), -1) / 255.0
Y_train = np.eye(10)[train.targets.numpy()]
X_test = test.data.numpy().astype(np.float32).reshape(len(test), -1) / 255.0
Y_test = np.eye(10)[test.targets.numpy()]

# Build the network
model = Network(
    layerSizes=[784, 128, 64, 10],
    activations=["relu", "relu", "softmax"],
    lossName="crossentropy",
    optimizer="adam",
    lambda_regularization=0.001,
)

model.ModelSummary()

# Train
model.Train(
    X_train, Y_train,
    X_test, Y_test,
    learningRate=0.01,
    epochs=10,
    batchSize=32,
)

model.TrainSummary()
model.PlotLossAccuracy()
model.PlotConfusionMatrix(X_test, Y_test, normalize=True)
```

## API overview

### Creating a network

```python
from lernet import Network

model = Network(
    layerSizes=[784, 128, 10],       # input → hidden → output
    activations=["relu", "softmax"],  # one per layer after the input
    lossName="crossentropy",          # "crossentropy" or "mse"
    optimizer="adam",                 # "sgd", "momentum", or "adam"
    lambda_regularization=0.001,
)
```

Empty strings in `activations` are filled automatically: ReLU for hidden layers, softmax (cross-entropy) or linear (MSE) for the output layer.

Softmax must be paired with cross-entropy; other output activations must use MSE.

### Training

```python
model.Train(
    X_train, Y_train,
    X_test, Y_test,
    learningRate=0.01,
    epochs=10,
    batchSize=32,
)
```

Each call to `Train()` appends a new run to the history, so you can compare different hyperparameters in one session.

### Inference

```python
outputs = model.Forward(X)       # raw layer outputs (softmax probabilities or regression values)
prediction = model.Predict(X)    # class index for the first sample
```

### Saving and loading

```python
model.Save("models/mnist_model.npz")

from lernet import FromFile, ListSavedModels

ListSavedModels("models/")
loaded = FromFile("mnist_model.npz", folderDir="models/")
```

### Utilities

```python
from lernet.utils import test_train_split, one_hot_encode, calculate_accuracy

X_train, Y_train, X_test, Y_test = test_train_split(X, Y, testSize=0.2)
Y_onehot, label_map = one_hot_encode(labels)
```

## Project structure

```
lernet/
├── pyproject.toml
├── README.md
└── src/
    └── lernet/
        ├── __init__.py      # public exports: Network, ListSavedModels
        ├── network.py       # Network class, training loop, plotting
        ├── layer.py         # single fully connected layer + backprop
        ├── activations.py   # activation functions and derivatives
        ├── losses.py        # loss functions and derivatives
        ├── optimizers.py    # SGD, Momentum, Adam
        └── utils.py         # data helpers, accuracy calculation
```

## How it works

1. **Forward pass** — each layer computes `z = X @ W + b`, then applies its activation.
2. **Loss** — cross-entropy or MSE is computed on the final output.
3. **Backward pass** — gradients flow backward layer by layer; softmax + cross-entropy uses the simplified combined gradient (`y_pred - y_true`).
4. **Update** — the chosen optimizer applies the gradients to weights and biases.

Weight initialization follows common heuristics: He init for ReLU, Xavier-style for sigmoid/tanh.

## License

MIT (or add your preferred license here)
