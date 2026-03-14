import numpy as np


def test_train_split(X, Y, testSize=0.2):
    numSamples = X.shape[0]
    indices = np.random.permutation(numSamples)

    testCount = int(numSamples * testSize)
    testIdx = indices[:testCount]
    trainIdx = indices[testCount:]

    X_train, Y_train = X[trainIdx], Y[trainIdx]
    X_test, Y_test = X[testIdx], Y[testIdx]

    return X_train, Y_train, X_test, Y_test

def calculate_accuracy(X_forward, Y_test, lossName):
    accuracy = None
    if(len(Y_test) > 0):
        if lossName == "crossentropy":
            #Classification: Use percent of time index of 1 is equal in pred and true
            yPred_test = X_forward
            preds = np.argmax(yPred_test, axis=1)
            true = np.argmax(Y_test, axis=1)
            accuracy = np.mean(preds == true)
        elif lossName == "mse":
            #Regression: use R^2 or inverse MSE as a “pseudo-accuracy”
            yPred_test = X_forward
            mse = np.mean((Y_test - yPred_test)**2)
            accuracy = 1 - mse / np.var(Y_test)

    return accuracy