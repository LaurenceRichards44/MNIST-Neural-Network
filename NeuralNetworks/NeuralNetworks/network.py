import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from layer import *
from utils import *
#from .layer import Layer
#from .utils import activationFunctionsDict, activationFunctionsDerivativeDict, lossFunctionsDict, lossFunctionsDerivativeDict

class Network():
    """
    A fully connected feedforward neural network implemented using NumPy.

    Features
    --------
    - Multiple hidden layers
    - Custom activation functions
    - Cross-entropy and MSE loss
    - Mini-batch gradient descent training
    - Model saving and loading
    - Training visualization tools

    Designed for educational purposes and experimentation.
    """
    def __init__(self, layerSizes=None, activations=None, lossName=None, optimizer=None):
        """
        Initialize a neural network.

        Parameters
        ----------
        layerSizes : list[int]
            List of integers defining the number of neurons in each layer,
            including input and output layers.

        activations : list[str]
            List of activation function names for each layer except the input layer.
            Must have length len(layerSizes) - 1.

        lossName : str
            Name of the loss function used for training.
            Supported values are defined in `lossFunctionsDict`.

        optimizer : str, optional
            Name of the optimizer used during training.
            (Currently unused but reserved for future expansion.)

        Raises
        ------
        ValueError
            If required parameters are missing or invalid.
        """
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
        if (self.activations[-1] == "softmax") != (self.lossName == "crossentropy"):
            raise ValueError("Softmax must be used with CrossEntropy, and other activations cannot use CrossEntropy")

    def InitializeLoss(self, lossName):
        """
        Initialize the loss function and its derivative.

        Parameters
        ----------
        lossName : str
            Name of the loss function to use.

        Notes
        -----
        The function and its derivative are retrieved from
        `lossFunctionsDict` and `lossFunctionsDerivativeDict`.
        """     
        self.lossFunction = lossFunctionsDict[lossName]
        self.lossFunctionDerivative = lossFunctionsDerivativeDict[lossName]

    def InitializeLayers(self, layerSizes, activations):
        """
        Create and initialize all layers of the neural network.

        Parameters
        ----------
        layerSizes : list[int]
            List of integers specifying the number of neurons in each layer.

        activations : list[str]
            List of activation function names corresponding to each layer.

        Raises
        ------
        ValueError
            If an activation function name is not supported.

        Notes
        -----
        Each layer contains:
        - Weight matrix
        - Bias vector
        - Activation function
        - Activation derivative
        """

        self.layers = []

        for i in range(len(layerSizes) - 1):
            # If user gave an empty string "", set default activation depending on the layer and the loss function
            if activations[i]:
                activationName = activations[i]
            elif i == len(layerSizes) - 2:
                activationName = "linear" if self.lossName == "mse" else "softmax"
            else:
                activationName = "relu"

            #Raise error if activation name is not supported
            if activationName not in activationFunctionsDict:
                raise ValueError(f"Activation '{activationName}' not supported. Available: {list(activationFunctionsDict.keys())}")

            #Get activation and corresponding derivative function according to activation name
            activationFunction = activationFunctionsDict[activationName]
            activationFunctionDerivative = activationFunctionsDerivativeDict.get(activationName, None)

            #Add layer
            self.layers.append(Layer(
                shape=(layerSizes[i], layerSizes[i + 1]),
                activation=activationFunction,
                activationDerivative=activationFunctionDerivative
            ))
            
    def Save(self, filename):
        """
        Save the trained model to disk.

        Parameters
        ----------
        filename : str
            Path to the file where the model should be saved.

        Notes
        -----
        The model is saved as a `.npz` file containing:

        - layer sizes
        - activation functions
        - loss function name
        - weights and biases for each layer
        - training history (learning rate, epochs, batch size)
        - loss history
        - accuracy history
        """
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
        """
        Load a saved model from disk.

        Parameters
        ----------
        filename : str
            Path to the `.npz` file containing the model.

        Notes
        -----
        This restores:
        - network architecture
        - weights and biases
        - loss function
        - training history
        """
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
        """
        Create a Network instance directly from a saved file.

        Parameters
        ----------
        filename : str
            Path to the saved model file.

        Returns
        -------
        Network
            A fully initialized network loaded from disk.
        """
        model = cls()
        model.Load(filename)
        return model
        
    def Forward(self, X):
        """
        Perform a forward pass through the neural network.

        Parameters
        ----------
        X : np.ndarray
            Input data with shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Output of the final layer.

        Notes
        -----
        This method returns the raw network outputs.

        - For **regression tasks**, this method should be used to obtain
        predicted numerical values.
        - For **classification tasks**, this method returns the predicted
        class probabilities (e.g. Softmax outputs).

        In classification problems, `Predict()` should usually be used
        instead to obtain the final class labels.
        """
        for layer in self.layers:
            X = layer.Forward(X)
        return X
    
    def Predict(self, X):
        """
        Generate predicted class labels for input data.

        Parameters
        ----------
        X : np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Array containing the predicted class index for each sample.

        Notes
        -----
        This method should be used for **classification tasks**.

        It performs a forward pass through the network and then returns
        the index of the highest output probability for each sample
        (using `argmax`).

        For **regression tasks**, use `Forward()` instead to obtain the
        predicted numerical outputs.
        """
        return np.argmax(self.Forward(X), axis=1)
    
    def Train(self, X, Y, learningRate=0.01, epochs=10, batchSize=32, testSize=0.2):
        """
        Train the neural network using mini-batch gradient descent.

        Parameters
        ----------
        X : np.ndarray
            Training inputs of shape (n_samples, n_features).

        Y : np.ndarray
            One-hot encoded target labels of shape (n_samples, n_classes).

        learningRate : float, default=0.01
            Learning rate used to update weights.

        epochs : int, default=10
            Number of training iterations over the dataset.

        batchSize : int, default=32
            Number of samples per training batch.

        testSize : float, default=0.2
            Fraction of the dataset reserved for testing.

        Notes
        -----
        Training process:
        1. Split data into train and test sets
        2. Shuffle training samples
        3. Perform forward pass
        4. Compute loss
        5. Backpropagate gradients
        6. Update weights and biases

        Training history (loss and accuracy) is stored in:

        - `self.lossesArray`
        - `self.accuracyArray`
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
        n = len(X_train)

        for epoch in range(epochs):
            #intitalize total loss and randomize batches
            totalLoss = 0
            indices = np.random.permutation(n)

            #Loop through batches for each epoch
            for i in range(0, n, batchSize):
                #Get batch X and Y
                batchIdx = indices[i:i+batchSize]
                x = X_train[batchIdx]
                y = Y_train[batchIdx]
                
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

            #Calculate accuracy if test data is provided
            if(len(X_test) > 0):
                if self.lossName == "crossentropy":
                    #Classification: Use percent of time index of 1 is equal in pred and true
                    yPred_test = self.Forward(X_test)
                    preds = np.argmax(yPred_test, axis=1)
                    true = np.argmax(Y_test, axis=1)
                    accuracy = np.mean(preds == true)
                    accuracies.append(accuracy)
                elif self.lossName == "mse":
                    #Regression: use R^2 or inverse MSE as a “pseudo-accuracy”
                    yPred_test = self.Forward(X_test)
                    mse = np.mean((Y_test - yPred_test)**2)
                    r2 = 1 - mse / np.var(Y_test)
                else:
                        accuracy = None
            #Append loss
            losses.append(loss)
            
            #Print progress
            if len(X_test) > 0:
                if self.lossName == "crossentropy":
                    print(f"Epoch {epoch+1:<5} Loss={loss:.4f}, Test Accuracy={accuracy*100:.2f}%", flush=True)
                elif self.lossName == "mse":
                    print(f"Epoch {epoch+1:<5} Loss={loss:.4f}, Test R^2={r2:.4f}", flush=True)
            else:
                print(f"Epoch {epoch+1:<5} Loss={loss:.4f}, Test data not provided. No accuracy available.", flush=True)
            
        #Append losses and accuracies
        if accuracies:
            self.accuracyArray.append(accuracies)
        self.lossesArray.append(losses)

    def Summary(self):
        """
        Print a summary of the neural network architecture.

        Displays:
        - layer index
        - input → output size
        - activation function
        - number of parameters

        Also prints the total parameter count.
        """
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
        """
        Plot training loss and accuracy over epochs.

        Parameters
        ----------
        splitters : bool, default=True
            If True, draw vertical lines separating multiple training runs.

        linearFit : bool, default=True
            If True, fit a linear regression to the later portion of the
            accuracy curve to estimate improvement rate.

        Notes
        -----
        Two plots are produced:

        1. Loss vs Epoch
        2. Accuracy vs Epoch

        Linear regression is applied to the last portion of each run
        to estimate learning trends.
        """
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