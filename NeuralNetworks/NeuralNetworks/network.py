import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from .layer import *
from .utils import *

class Network():
    def __init__(self, layerSizes=None, activations=None, lossName="CrossEntropy", optimizer=None):
        if lossName not in lossFunctionsDict:
            raise ValueError(f"Loss '{lossName}' not supported. Available: {list(lossFunctionsDict.keys())}")
        
        for activation in activations:
            if activation not in activationFunctionsDict:
                raise ValueError(f"Activation '{activation}' not supported. Available: {list(activationFunctionsDict.keys())}")

        if lossName == "CrossEntropy":
            outputActivation = activations[-1] if activations else None
            print(outputActivation)
            if outputActivation not in ["Softmax", None]:
                raise ValueError(f"Activation '{outputActivation}' not supported with cross entropy")
    
        if lossName == "mse" and activations and activations[-1] == "Softmax":
            print("Warning: Using MSE with Softmax output; consider cross-entropy for classification")

        self.dims = layerSizes
        self.lossName = lossName
        self.lossFunction = None
        self.lossFunctionDerivative = None

        self.InitializeLoss(lossName)
        self.InitializeLayers(self.dims, activations)

        self.learningRate = []
        self.epochs = []
        self.batchSize = []
        
        self.accuracyArray = []
        self.lossesArray = []

    @classmethod
    def FromFile(cls, filename = r"C:\Users\laure\OneDrive\Desktop\coding\Python\Github MNIST\MNIST-Neural-Network\NeuralNetworks\models\best.npz"):
        model = cls()
        model.Load(filename)
        return model

    def InitializeLoss(self, lossName):        
        self.lossFunction = lambda y, y_hat: lossFunctionsDict[lossName](y, y_hat)
        self.lossFunctionDerivative = lambda y, y_hat: lossFunctionsDerivativeDict[lossName](y, y_hat)

    def InitializeLayers(self, layerSizes, activations):
        self.layers = []

        if layerSizes is not None:
            for i in range(len(layerSizes)-1):
                activationFunction = None
                activationFunctionDerivative = None

                if activations:
                    name = activations[i]

                    if name and name not in activationFunctionsDict:
                        raise ValueError(f"Activation '{name}' not supported")
                    
                    activationFunction = activationFunctionsDict.get(name, ReLU if i != len(layerSizes)-2 else Softmax)
                    activationFunctionDerivative = activationFunctionsDerivativeDict.get(name, activationFunctionsDerivativeDict["ReLU"] if i != len(layerSizes)-2 else activationFunctionsDerivativeDict["Softmax"])

                if i == len(layerSizes)-2 and activationFunction is None:
                    activationFunction = Softmax

                self.layers.append(Layer((layerSizes[i], layerSizes[i+1]), activationFunction, activationFunctionDerivative))
            
    def Save(self, filename):
        modelData = {}

        modelData["dims"] = self.dims

        for i, layer in enumerate(self.layers):
            modelData[f"w{i}"] = layer.weights
            modelData[f"b{i}"] = layer.biases

            modelData["lr"] = np.array(self.learningRate, dtype=object)
            modelData["e"] = np.array(self.epochs, dtype=object)
            modelData["bs"] = np.array(self.batchSize, dtype=object)

            modelData["a"] = np.array(self.accuracyArray, dtype=object)
            modelData["l"] = np.array(self.lossesArray, dtype=object)

        np.savez(filename, **modelData)
        print(f"Model saved to {filename}")

    def Load(self, filename):
        modelData = np.load(filename, allow_pickle=True)

        self.dims = modelData["dims"]
        self.InitializeLayers(self.dims)

        for i, layer in enumerate(self.layers):
            layer.weights = modelData[f"w{i}"]
            layer.biases = modelData[f"b{i}"]
            
        self.learningRate = modelData[f"lr"].tolist()
        self.epochs = modelData[f"e"].tolist()
        self.batchSize = modelData[f"bs"].tolist()
        
        self.accuracyArray = modelData["a"].tolist()
        self.lossesArray = modelData["l"].tolist() 

        print(f"Model loaded from \"{filename}\"")
    
    def Forward(self, X):
        for layer in self.layers:
            X = layer.Forward(X)
        return X
    
    def Predict(self, X):
        return np.argmax(self.Forward(X), axis=1)
    
    def Train(self, X, Y, X_test, Y_test, learningRate, epochs, batchSize):
        self.learningRate.append(learningRate)
        self.epochs.append(epochs)
        self.batchSize.append(batchSize)
        
        losses = []
        accuracies = []
        n = len(X)

        for epoch in range(epochs):
            totalLoss = 0
            indices = np.random.permutation(n)

            for i in range(0, n, batchSize):
                batchIdx = indices[i:i+batchSize]
                x = X[batchIdx]
                y = Y[batchIdx]
                
                y_hat = self.Forward(x)
                
                lossValue = self.lossFunction(y, y_hat)
                grad = self.lossFunctionDerivative(y, y_hat)
                
                for layer in reversed(self.layers):
                    grad = layer.Backward(grad, learningRate)
                
                totalLoss += lossValue
        
            loss = totalLoss / n
            if self.lossName == "CrossEntropy":
                preds = np.argmax(self.Forward(X_test), axis=1)
                true = np.argmax(Y_test, axis=1)
                accuracy = np.mean(preds == true)
                accuracies.append(accuracy)
            else:
                accuracy = None

            losses.append(loss)
            
            print(f"Epoch {epoch+1}: Loss={loss:.4f}  Test Accuracy={accuracy*100:.2f}%")
            
        if accuracies:
            self.accuracyArray.append(accuracies)
        self.lossesArray.append(losses)

    def PlotLossAccuracy(self, splitters : bool = True, linearFit : bool = True):
        losses = np.concatenate(self.lossesArray)
        accuracy = np.concatenate(self.accuracyArray)

        fig, ax = plt.subplots(1, 2, figsize=(12,6))

        midPercent = 0.8

        #===Loss===
        ax[0].plot(losses, color='red', linestyle='-', label='Model Loss')
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Average Loss")
        ax[0].legend()

        if splitters:
            offset = 0
            for i, run in enumerate(self.lossesArray):
                ax[0].axvline(offset, color='grey', linestyle='--') if i > 0 else None
                offset += len(run)

        #===Accuracy===
        ax[1].set_ylim(0, 1)
        ax[1].set_yticks(np.arange(0, 1.1, 0.1))
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Percentage Accuracy")
        ax[1].legend()

        ax[1].plot(accuracy, color='green', linestyle='-', label='Model Accuracy')

        if splitters:
            offset = 0
            for i, run in enumerate(self.lossesArray):
                ax[1].axvline(offset, color='grey', linestyle='--') if i > 0 else None
                offset += len(run)

        if linearFit:
            offset = 0
            for i, run in enumerate(self.accuracyArray):
                run = np.array(run)
                runLength = len(run)
            
                startLocal = int(runLength * midPercent)
                xFit = np.arange(offset + startLocal, offset + runLength).reshape(-1,1)
                yFit = run[startLocal:]
                
                fit = LinearRegression().fit(xFit, yFit)
                
                xFull = np.arange(offset, offset + runLength).reshape(-1,1)
                yPredFull = fit.predict(xFull)
                
                ax[1].plot(xFull, yPredFull, color='black', linestyle='--', label=f'Fit {i+1} ({fit.coef_[0]*10000:.2f}% per 100 epochs)')

                offset += runLength

        plt.tight_layout()
        plt.show()