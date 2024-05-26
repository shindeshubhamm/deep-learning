import numpy as np


class Module:

    def forward(self, *args, **kwargs):
        pass


class Network(Module):

    def __init__(self, layers=None):
        if layers is not None:
            self.layers = layers
        else:
            self.layers = []

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def add_layer(self, layer):
        self.layers.append(layer)


class LinearLayer(Module):

    def __init__(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        output = np.dot(self.W, x) + self.b
        return output


class Sigmoid(Module):

    def forward(self, x):
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid


class ReLU(Module):

    def forward(self, x):
        relu = np.maximum(x, 0)
        return relu


class Loss(Module):

    def forward(self, prediction, target):
        pass


class MSE(Loss):

    def forward(self, prediction, target):
        mse = np.mean((prediction - target) ** 2)
        return mse


class CrossEntropyLoss(Loss):

    def forward(self, prediction, target):
        prediction = prediction
        softmax = np.exp(prediction) / np.sum(np.exp(prediction))
        ce = -np.log(softmax)
        return ce
