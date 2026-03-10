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

activationFunctionsDict = {
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "softmax": Softmax,
}

#===Activation Functions Derivatives===
def ReluDerivative(x):
    return (x > 0).astype(float)

def SigmoidDerivative(x):
    s = Sigmoid(x)
    return s * (1 - s)

def TanhDerivative(x):
    return 1 - np.tanh(x) ** 2

activationFunctionsDerivativeDict = {
    "relu": ReluDerivative,
    "sigmoid": SigmoidDerivative,
    "tanh": TanhDerivative,
    "softmax": None
    #Softmax is none because the differentiation is handled
    #when combined with Categorical cross entropy as the derivative
    # is much simpler
}

#===Loss Functions===
def CrossEntropyLoss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-12), axis=1))

def MSELoss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

lossFunctionsDict = {
    "crossentropy": CrossEntropyLoss,
    "mse": MSELoss
}

#===Loss Functions Derivatives===
def CrossEntropyDerivative(y_true, y_pred):
    return y_pred - y_true

def MSELossDerivative(y_true, y_pred):
    return 2 / y_true.shape[0] * (y_pred - y_true)

lossFunctionsDerivativeDict = {
    "crossentropy": CrossEntropyDerivative,
    "mse": MSELossDerivative
}