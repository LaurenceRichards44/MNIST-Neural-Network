import numpy as np
#from .utils import *
from utils import *

<<<<<<< HEAD
class Layer:
    def __init__(self, shape, activation=None, activationDerivative=None):
        """
        shape: tuple (input_size, output_size)
        activation: activation function
        activationDerivative: derivative of activation function
        """
        # Choose proper weight initialization
        if activation is None or activation == Linear or activation == Linear:
            scale = 0.01
        else:
            scale = np.sqrt(2 / shape[0])
=======
class Layer():
    def __init__(self, shape, activation = None, activationDerivative=None):
        self.weights = np.random.randn(*shape) * (0.01 if activation == Softmax else np.sqrt(2 / shape[0]))
        self.biases = np.zeros((1,shape[1]))
>>>>>>> ee3109b309db07745674392b3c66063d2b4d4465

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
<<<<<<< HEAD
        da = self.activationDerivative(self.z)
=======
        da = self.activationDerivative(self.z) if self.activationDerivative != None else 1
>>>>>>> ee3109b309db07745674392b3c66063d2b4d4465
        dz = gradOut * da

        self.dw = np.dot(self.input.T, dz) / dz.shape[0]
        self.db = np.sum(dz, axis=0, keepdims=True) / dz.shape[0]

        gradInput = np.dot(dz, self.weights.T)
        return gradInput

    def Update(self, learningRate):
        self.weights -= self.dw * learningRate
        self.biases -= self.db * learningRate


    def Print(self):
        return f"Weights: {self.weights.shape} | Biases: {self.biases.shape} | Activation: {self.activation.__name__} | Activation Derivative: {self.activationDerivative.__name__}"