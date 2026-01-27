import numpy as np
import pandas as pd


class ArtificialNeuralNetwork:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray):
        self.x_train = x_train
        self.y_train = y_train

        self.weights = {}
        self.preactivations = {}
        self.activations = {}

        self.data_size: tuple = np.shape(x_train)

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def dense(self, size: int):
        self.weights[len(self.weights) + 1] = np.random.randn(self.data_size[1], size)

    def train(self):
        self.weights[len(self.weights) + 1] = np.random.randn(
            np.shape(list(self.weights.values())[-1])[1], 1
        )
