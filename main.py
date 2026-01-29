import sys
import time

import numpy as np
import pandas as pd


class ArtificialNeuralNetwork:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray):
        self.x_train = x_train
        self.y_train = y_train.reshape(-1, 1)

        self.weights = {}
        self.biases = {}
        self.preactivations = {}
        self.activations = {}

        self.gradient_weights = {}
        self.gradient_biases = {}
        self.delta: np.ndarray = []

        self.data_size: tuple = np.shape(x_train)

    def relu(self, x: np.ndarray):
        return np.maximum(0, x)

    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def binary_cross_entropy(
        self, y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-12
    ) -> float:
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return float(np.mean(loss))

    def dense(self, size: int):
        if len(self.weights) <= 0:
            self.weights[len(self.weights) + 1] = np.random.randn(
                self.data_size[1], size
            )
            self.biases[len(self.biases) + 1] = np.zeros((1, size))

        else:
            self.weights[len(self.weights) + 1] = np.random.randn(
                np.shape(list(self.weights.values())[-1])[1], size
            )
            self.biases[len(self.biases) + 1] = np.zeros((1, size))

    def forward(self, x: np.ndarray):
        self.activations[0] = x
        for i in range(len(self.weights)):
            self.preactivations[i + 1] = (
                self.activations[i] @ self.weights[i + 1] + self.biases[i + 1]
            )
            if i == len(self.weights) - 1:
                self.activations[i + 1] = self.sigmoid(self.preactivations[i + 1])
                continue
            self.activations[i + 1] = self.relu(self.preactivations[i + 1])

    def backward(self):
        dy = self.activations[len(self.activations) - 1] - self.y_train
        self.delta = dy
        for i in range(len(self.weights), 0, -1):
            dw = self.activations[i - 1].T @ self.delta
            self.gradient_weights[i] = dw

            db = np.sum(self.delta, axis=0, keepdims=True)
            self.gradient_biases[i] = db
            dx = self.delta @ self.weights[i].T
            if i == 1:
                continue
            self.delta = dx * self.relu_derivative(self.preactivations[i - 1])

    def update_params(self, lr: float):
        for i in range(len(self.weights)):
            self.weights[i + 1] -= lr * self.gradient_weights[i + 1]
            self.biases[i + 1] -= lr * self.gradient_biases[i + 1]

    def _train_visualization(self, loss: float, acc, epochs: int, i: int):
        # start timer on first epoch
        if not hasattr(self, "_train_start"):
            self._train_start = time.time()
        elapsed = time.time() - self._train_start
        avg_per_epoch = elapsed / (i + 1)
        remaining = avg_per_epoch * (epochs - (i + 1))
        eta_min = int(remaining // 60)
        eta_sec = int(remaining % 60)

        # simple progress bar
        bar_len = 40
        filled = int((i + 1) / epochs * bar_len)
        bar = "#" * filled + "-" * (bar_len - filled)

        # print single-line dynamic status
        sys.stdout.write(
            f"\rEpoch {i + 1}/{epochs} [{bar}] loss={loss:.4f} acc={acc * 100:5.1f}% ETA={eta_min:02d}:{eta_sec:02d}"
        )
        sys.stdout.flush()
        if i == epochs - 1:
            print()

    def train(self, epochs: int, lr: float):
        if len(self.weights) <= 0:
            raise RuntimeError("Initial weight first (dense)")

        for i in range(epochs):
            self.forward(self.x_train)
            # loss & accuracy
            y_pred = self.activations[len(self.activations) - 1].ravel()
            loss = self.binary_cross_entropy(self.y_train.ravel(), y_pred)
            acc = np.mean((y_pred >= 0.5) == self.y_train.ravel())

            self._train_visualization(loss, acc, epochs, i)

            self.backward()
            self.update_params(lr)
            time.sleep(0.001)

    def prediction(self, x: np.ndarray):
        self.forward(x)
        y_pred = self.activations[len(self.activations) - 1]
        return y_pred


df = pd.read_csv("./dataset/XOR_dataset.csv")
x_train = df.iloc[:, 0:-1].values
y_train = df.iloc[:, -1].values

x_test = np.array([[0, 0], [1, 1]])
y_test = np.array([[0], [0]])
model = ArtificialNeuralNetwork(np.array(x_train), np.array(y_train))

model.dense(8)
model.dense(4)
model.dense(1)
model.train(1000, 0.01)

y_pred = model.prediction(x_test)
print(f"Y prediction: \n{y_pred}")
print("==")
print(f"Y actual: \n{y_test}")
