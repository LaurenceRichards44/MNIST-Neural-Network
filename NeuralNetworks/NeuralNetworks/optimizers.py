import numpy as np

class Optimizer:
    def step(self, layers):
        raise NotImplementedError
    
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def step(self, layers):
        for layer in layers:
            layer.weights -= self.lr * layer.dw
            layer.biases -= self.lr * layer.db

class Momentum(Optimizer):

    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.vw = []
        self.vb = []
        self.initialized = False

    def step(self, layers):

        if not self.initialized:
            for layer in layers:
                self.vw.append(np.zeros_like(layer.weights))
                self.vb.append(np.zeros_like(layer.biases))
            self.initialized = True

        for i, layer in enumerate(layers):

            self.vw[i] = self.beta * self.vw[i] + (1 - self.beta) * layer.dw
            self.vb[i] = self.beta * self.vb[i] + (1 - self.beta) * layer.db

            layer.weights -= self.lr * self.vw[i]
            layer.biases -= self.lr * self.vb[i]

class Adam(Optimizer):

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.mw = []
        self.vw = []

        self.mb = []
        self.vb = []

        self.t = 0
        self.initialized = False


    def step(self, layers):

        if not self.initialized:
            for layer in layers:
                self.mw.append(np.zeros_like(layer.weights))
                self.vw.append(np.zeros_like(layer.weights))

                self.mb.append(np.zeros_like(layer.biases))
                self.vb.append(np.zeros_like(layer.biases))

            self.initialized = True

        self.t += 1

        for i, layer in enumerate(layers):

            # first moment
            self.mw[i] = self.beta1 * self.mw[i] + (1 - self.beta1) * layer.dw
            self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * layer.db

            # second moment
            self.vw[i] = self.beta2 * self.vw[i] + (1 - self.beta2) * (layer.dw ** 2)
            self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * (layer.db ** 2)

            # bias correction
            mw_hat = self.mw[i] / (1 - self.beta1 ** self.t)
            vw_hat = self.vw[i] / (1 - self.beta2 ** self.t)

            mb_hat = self.mb[i] / (1 - self.beta1 ** self.t)
            vb_hat = self.vb[i] / (1 - self.beta2 ** self.t)

            layer.weights -= self.lr * mw_hat / (np.sqrt(vw_hat) + self.eps)
            layer.biases -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)