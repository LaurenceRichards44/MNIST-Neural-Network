import numpy as np

#===Activation Functions===
def ReLU(x):
    return np.maximum(0, x)

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Tanh(x):
    return np.tanh(x)

def Softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)

#===Loss Functions===
def CrossEntropyLoss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-12))

def mseLoss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def maeLoss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

activationFunctionsDict = {
    "ReLU": ReLU,
    "Sigmoid": Sigmoid,
    "Tanh": Tanh,
    "Softmax": Softmax,
}

activationFunctionsDerivativeDict = {
    "ReLU": lambda x: (x > 0).astype(float),
    "Sigmoid": lambda x: Sigmoid(x) * (1 - Sigmoid(x)),
    "Tanh": lambda x: 1 - np.tanh(x) ** 2,
    "Softmax": None
}

lossFunctionsDict = {
    "CrossEntropy": lambda y, y_hat: -np.sum(y * np.log(y_hat + 1e-12)),
    "MSE": lambda y, y_hat: np.mean((y - y_hat) ** 2)
}

lossFunctionsDerivativeDict = {
    "CrossEntropy": lambda y, y_hat: (y_hat - y),
    "MSE": lambda y, y_hat: (y_hat - y) / y.shape[0]
}