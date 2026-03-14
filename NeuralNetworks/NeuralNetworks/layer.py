import numpy as np
from .activations import *

class Layer:
    def __init__(self, shape, activation=None, activationDerivative=None, lambda_regularization=0.001):
        """
        shape: tuple (input_size, output_size)
        activation: activation function
        activationDerivative: derivative of activation function
        """
        # Choose proper weight initialization
        if activation == ReLU:
            scale = np.sqrt(2 / shape[0])
        elif activation in (Sigmoid, Tanh):
            scale = np.sqrt(1 / shape[0])
        else:
            scale = 0.01

        self.lambda_regularization = lambda_regularization


        self.weights = np.random.randn(*shape) * scale
        self.biases = np.zeros((1, shape[1]))
        
        self.activation = activation
        self.activationDerivative = activationDerivative
        
    def Forward(self, X):
        self.input = X

        self.z = np.dot(self.input, self.weights) + self.biases
        self.a = self.z if self.activation is None else self.activation(self.z)

        return self.a
    
    def ComputeGradients(self, gradOut):
        da = self.activationDerivative(self.z) if self.activationDerivative is not None else 1
        dz = gradOut * da

        self.dw = np.dot(self.input.T, dz) / dz.shape[0] + self.lambda_regularization * self.weights
        self.db = np.sum(dz, axis=0, keepdims=True) / dz.shape[0]

        gradInput = np.dot(dz, self.weights.T)
        return gradInput

    def Update(self, learningRate):
        self.weights -= self.dw * learningRate
        self.biases -= self.db * learningRate