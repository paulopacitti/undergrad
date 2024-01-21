from abc import ABC, abstractmethod
import numpy as np


class BaseOp(ABC):
    @abstractmethod
    def __call__(self, X: np.array):
        pass

    @abstractmethod
    def grad(self, X: np.array):
        pass


class BaseLoss(ABC):
    @abstractmethod
    def __call__(self, Y: np.array, Y_pred: np.array):
        pass

    @abstractmethod
    def grad(self, Y: np.array, Y_pred: np.array):
        pass


class ReLU(BaseOp):
    def __call__(self, X: np.array):
        return np.maximum(np.zeros_like(X), X)

    def grad(self, X: np.array):
        return np.where(X >= 0, 1, 0)


class LeakyReLU(BaseOp):
    def __call__(self, X: np.array):
        return np.maximum(0.01 * X, X)

    def grad(self, X: np.array):
        return np.where(X >= 0, 1, 0.01)


class Softmax(BaseOp):
    def __call__(self, X: np.array):
        unique_values = dict(axis=1, keepdims=True)

        # subtracts the maximum value of each row of the input array, to prevent overflow of np.exp
        X_rel = X - X.max(**unique_values)

        exp_X_rel = np.exp(X_rel)
        return exp_X_rel / np.sum(exp_X_rel, axis=1, keepdims=True)

    def grad(self, X: np.array):
        return 1  # discard these gradients


class CrossEntropy(BaseLoss):
    def __call__(self, Y: np.array, Y_pred: np.array):
        epsilon = 1e-8
        return -np.sum(Y * np.log(Y_pred + epsilon)) / Y.shape[0]

    def grad(self, Y: np.array, Y_pred: np.array):
        return Y_pred - Y
