import numpy as np
from .utils import *

class Layer():
    def __init__(self, shape, activation = None, activationDerivative=None):
        if activation == Softmax:
            self.weights = np.random.randn(*shape) * 0.01
        else:
            self.weights = np.random.randn(*shape) * np.sqrt(2 / shape[0])
        self.biases = np.zeros((1,shape[1]))
        self.activation = activation
        self.activationDerivative = activationDerivative
        
    def Forward(self, X):
        self.input = X

        self.z = np.dot(self.input, self.weights) + self.biases
        self.a = self.z if self.activation is None else self.activation(self.z)

        return self.a
    
    def Backward(self, gradOut, learningRate):
        da = self.activationDerivative(self.z) if self.activationDerivative else 1
        dz = gradOut * da

        dw = np.dot(self.input.T, dz) / dz.shape[0]
        db = np.sum(dz, axis=0, keepdims=True) / dz.shape[0]

        self.weights -= dw * learningRate
        self.biases -= db * learningRate

        gradInput = np.dot(dz, self.weights.T)
        return gradInput