import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from layer import *
from utils import *
#from .layer import Layer
#from .utils import activationFunctionsDict, activationFunctionsDerivativeDict, lossFunctionsDict, lossFunctionsDerivativeDict

class Network():
    def __init__(self, layerSizes=None, activations=None, lossName=None, optimizer=None):
        self.layerSizes = layerSizes

        self.lossName = lossName.lower() if lossName != None else lossName
        self.activations = [activation.lower() for activation in activations]
        self.optimizer = optimizer.lower() if optimizer != None else optimizer

        self.lossFunction = None
        self.lossFunctionDerivative = None

        self.InitializeLoss(self.lossName)
        self.InitializeLayers(self.layerSizes, self.activations)

        self.learningRate = []
        self.epochs = []
        self.batchSize = []
        
        self.accuracyArray = []
        self.lossesArray = []

        #Raise error if any parameters are not provided
        if self.layerSizes is None:
            raise ValueError("layerSizes must be provided")
        #---UPDATE: ADD ABILITY TO FILL WILL RELU AND SOFTMAX IF ACTIVATIONS NOT PROVIDED
        if self.activations is None:
            raise ValueError("activations must be provided")
        if self.lossName is None:
            raise ValueError("lossName must be provided")
        
        #Raise error if activation list has incorrect length or lossName is not supported
        if len(self.activations) != len(self.layerSizes) - 1:
            raise ValueError(f"activations list must have exactly {len(self.layerSizes) - 1} elements")
        if self.lossName not in lossFunctionsDict:
            raise ValueError(f"Loss '{self.lossName}' not supported. Available: {list(lossFunctionsDict.keys())}")

        #Raise error if final activation is not supported with loss
        if (self.activations[-1] == "Softmax") != (lossName == "crossentropy"):
            raise ValueError("Softmax must be used with CrossEntropy, and other activations cannot use CrossEntropy")

    def PrintLayerInfo(self):
        for layer in self.layers:
            print(f"Layer {self.layers.index(layer)}: {layer.Print()}")

    def InitializeLoss(self, lossName):        
        self.lossFunction = lossFunctionsDict[lossName]
        self.lossFunctionDerivative = lossFunctionsDerivativeDict[lossName]

    def InitializeLayers(self, layerSizes, activations):
        """
        Initialize the network layers.

        Parameters
        ----------
        layerSizes : list[int]
            List of integers representing number of neurons in each layer.
        activations : list[str]
            List of activation names for each layer (length must be len(layerSizes)-1).
            Use "" to default to ReLU for that layer.
        """

        self.layers = []

        for i in range(len(layerSizes) - 1):
            #Get activation name, if not provided default to ReLU
            acttivationName = activations[i].lower() if activations[i] else "relu"

            #Raise error if activation name is not supported
            if acttivationName not in activationFunctionsDict:
                raise ValueError(f"Activation '{acttivationName}' not supported. Available: {list(activationFunctionsDict.keys())}")

            #Get activation and corresponding derivative function according to activation name
            activationFunction = activationFunctionsDict[acttivationName]
            activationFunctionDerivative = activationFunctionsDerivativeDict.get(acttivationName, None)

            #Add layer
            self.layers.append(Layer(
                shape=(layerSizes[i], layerSizes[i + 1]),
                activation=activationFunction,
                activationDerivative=activationFunctionDerivative
            ))
            
    def Save(self, filename):
        modelData = {}

        modelData["layersizes"] = self.layerSizes

        modelData["activations"] = self.activations
        modelData["lossname"] = self.lossName

        for i, layer in enumerate(self.layers):
            modelData[f"weights: {i}"] = layer.weights
            modelData[f"biases: {i}"] = layer.biases

            modelData["learningrate"] = np.array(self.learningRate, dtype=object)
            modelData["epochs"] = np.array(self.epochs, dtype=object)
            modelData["batchsize"] = np.array(self.batchSize, dtype=object)

            modelData["accuracies"] = np.array(self.accuracyArray, dtype=object)
            modelData["losses"] = np.array(self.lossesArray, dtype=object)

        np.savez(filename, **modelData)
        print(f"Model saved to {filename}")

    def Load(self, filename):
        modelData = np.load(filename, allow_pickle=True)

        self.layerSizes = modelData["layersizes"]

        self.activations = modelData["activations"]
        self.lossName = modelData["lossname"]

        self.InitializeLoss(self.lossName)
        self.InitializeLayers(self.layerSizes, self.activations)

        for i, layer in enumerate(self.layers):
            layer.weights = modelData[f"weights: {i}"]
            layer.biases = modelData[f"biases: {i}"]
            
        self.learningRate = modelData[f"learningrate"].tolist()
        self.epochs = modelData[f"epochs"].tolist()
        self.batchSize = modelData[f"batchsize"].tolist()
        
        self.accuracyArray = modelData["accuracies"].tolist()
        self.lossesArray = modelData["losses"].tolist() 

        print(f"Model loaded from \"{filename}\"")

    @classmethod
    def FromFile(cls, filename = r"C:\Users\laure\OneDrive\Desktop\coding\Python\Github MNIST\MNIST-Neural-Network\NeuralNetworks\models\best.npz"):
        model = cls()
        model.Load(filename)
        return model
        
    def Forward(self, X):
        for layer in self.layers:
            X = layer.Forward(X)
        return X
    
    def Predict(self, X):
        return np.argmax(self.Forward(X), axis=1)
    
    def Train(self, X, Y, learningRate=0.01, epochs=10, batchSize=32, testSize=0.2):
        """
        Train the neural network on the given dataset.

        Parameters
        ----------
        X : np.ndarray
            Input features, shape (n_samples, n_features)
        Y : np.ndarray
            One-hot encoded labels, shape (n_samples, n_classes)
        learningRate : float, default=0.01
            Learning rate for weight updates
        epochs : int, default=10
            Number of training epochs
        batchSize : int, default=32
            Size of mini-batches
        testSize : float, default=0.2
            Fraction of data to use as test set for accuracy evaluation

        Returns
        -------
        None
            Updates the network's weights in place and records training loss and accuracy.
        
        Notes
        -----
        - Uses mini-batch gradient descent.
        - Supports CrossEntropy and MSE loss functions.
        - Accuracy calculation requires X_test and Y_test to be provided.
        - Loss and accuracy for each epoch are stored in `self.lossesArray` and `self.accuracyArray`.
        """
        
        #Store hyperparameters
        self.learningRate.append(learningRate)
        self.epochs.append(epochs)
        self.batchSize.append(batchSize)

        #Create training and test data based on test-train-split
        numSamples = X.shape[0]
        indices = np.arange(numSamples)

        testCount = int(numSamples * testSize)
        testIdx = indices[:testCount]
        trainIdx = indices[testCount:]

        X_train, Y_train = X[trainIdx], Y[trainIdx]
        X_test, Y_test = X[testIdx], Y[testIdx]

        #Initialize arrays for loss and accuracy
        losses = []
        accuracies = []
        n = len(X)

        for epoch in range(epochs):
            #intitalize total loss and randomize batches
            totalLoss = 0
            indices = np.random.permutation(n)

            #Loop through batches for each epoch
            for i in range(0, n, batchSize):
                #Get batch X and Y
                batchIdx = indices[i:i+batchSize]
                x = X[batchIdx]
                y = Y[batchIdx]
                
                #Forward pass
                yPred = self.Forward(x)
                
                #Calculate loss and loss function gradient to prepare for Vanilla SGD
                lossValue = self.lossFunction(y, yPred)
                grad = self.lossFunctionDerivative(y, yPred)
                
                #Calculate gradients for each layer
                for layer in reversed(self.layers):
                    grad = layer.ComputeGradients(grad)
                
                #Update gradients for each layer
                for layer in self.layers:
                    layer.Update(learningRate)

                #Update total loss
                totalLoss += lossValue

            #Calculate average loss for epoch
            loss = totalLoss / n

            #Calculate accuracy
            if self.lossName == "crossentropy":
                #Classification: Use percent of time index of 1 is equal in pred and true
                yPred_test = self.Forward(X_test)
                preds = np.argmax(yPred_test, axis=1)
                true = np.argmax(Y_test, axis=1)
                accuracy = np.mean(preds == true)
                accuracies.append(accuracy)
            elif self.lossName.lower() == "mse":
                #Regression: use R^2 or inverse MSE as a “pseudo-accuracy”
                yPred_test = self.Forward(X_test)
                mse = np.mean((Y_test - yPred_test)**2)
                accuracy = 1 - mse / np.var(Y_test)
            else:
                accuracy = None

            #Append loss
            losses.append(loss)
            
            #Print progress
            print(f"Epoch {epoch+1}: Loss={loss:.4f}  Test Accuracy={accuracy*100:.2f}%")
            
        #Append losses and accuracies
        if accuracies:
            self.accuracyArray.append(accuracies)
        self.lossesArray.append(losses)

    def Summary(self):
        print("Model Summary")
        print("="*50)
        print(f"{'Layer':<10} {'Input → Output':<20} {'Activation':<15} {'Params':<10}")
        print("-"*50)

        total_params = 0
        for i, layer in enumerate(self.layers):
            input_size, output_size = layer.weights.shape
            params = input_size * output_size + output_size  # weights + biases
            total_params += params
            activation_name = layer.activation.__name__ if layer.activation else "None"
            print(f"{i:<10} {input_size}→{output_size:<15} {activation_name:<15} {params:<10}")

        print("-"*50)
        print(f"Total parameters: {total_params}")
        print("="*50)

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
