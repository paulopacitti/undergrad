from abc import ABC, abstractmethod
import numpy as np


class BaseOp(ABC):
    @abstractmethod
    def __call__(self, X):
        pass

    @abstractmethod
    def grad(self, X):
        pass


class ReLU(BaseOp):
    def __call__(self, X):
        return np.maximum(np.zeros_like(X), X)

    def grad(self, X):
        return np.where(X >= 0, 1, 0)


class LeakyReLU(BaseOp):
    def __call__(self, X):
        return np.maximum(0.01 * X, X)

    def grad(self, X):
        return np.where(X >= 0, 1, 0.01)


class Softmax(BaseOp):
    def __call__(self, X):
        unique_values = dict(axis=1, keepdims=True)

        # subtracts the maximum value of each row of the input array, to prevent overflow of np.exp
        X_rel = X - X.max(**unique_values)

        exp_X_rel = np.exp(X_rel)
        return exp_X_rel / np.sum(exp_X_rel, axis=1, keepdims=True)

    def grad(self, X):
        return 1  # discard these gradients


class CrossEntropy(BaseOp):
    def __call__(self, Y, Y_pred):
        epsilon = 1e-8
        return -np.sum(Y * np.log(Y_pred + epsilon)) / Y.shape[0]

    def grad(self, Y, Y_pred):
        return Y_pred - Y
