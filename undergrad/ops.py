from abc import ABC, abstractmethod
import numpy as np


class BaseOp(ABC):
    @abstractmethod
    def __call__(self, X: np.array):
        pass

    @abstractmethod
    def grad(self, X: np.array):
        pass


class BaseLossFunc(ABC):
    @abstractmethod
    def __call__(self, Y: np.array, Y_pred: np.array):
        pass

    @abstractmethod
    def grad(self, Y: np.array, Y_pred: np.array):
        pass


class ReLU(BaseOp):
    """
   The ReLU (Rectified Linear Unit) activation function. It is an element-wise operation that replaces negative values with zeros. It is commonly used in neural networks to introduce non-linearity.

    Methods
    -------
    __call__(X: np.array) -> np.array:
        Applies the ReLU activation function element-wise to the input array.

    grad(X: np.array) -> np.array:
        Computes the gradient of the ReLU activation function.

    Examples
    --------
    >>> op = ReLU()
    >>> X = np.array([-1, 0, 1])
    >>> op(X)
    array([0, 0, 1])
    >>> op.grad(X)
    array([0, 1, 1])
    """

    def __call__(self, X: np.array) -> np.array:
        return np.maximum(np.zeros_like(X), X)

    def grad(self, X: np.array) -> np.array:
        return np.where(X >= 0, 1, 0)


class LeakyReLU(BaseOp):
    """
    The LeakyReLU (Leaky Rectified Linear Unit) activation function. It is an element-wise operation that replaces negative values with a small positive value. It is commonly used in neural networks to introduce non-linearity and mitigate the dying ReLU problem.

    Methods
    -------
    __call__(X: np.array) -> np.array:
        Applies the LeakyReLU activation function element-wise to the input array.

    grad(X: np.array) -> np.array:
        Computes the gradient of the LeakyReLU activation function.

    Examples
    --------
    >>> op = LeakyReLU()
    >>> X = np.array([-1, 0, 1])
    >>> op(X)
    array([-0.01,  0.  ,  1.  ])
    >>> op.grad(X)
    array([0.01, 1.  , 1.  ])
    """

    def __call__(self, X: np.array) -> np.array:
        return np.maximum(0.01 * X, X)

    def grad(self, X: np.array) -> np.array:
        return np.where(X >= 0, 1, 0.01)


class Softmax(BaseOp):
    """
    Softmax activation function. It is used in the output layer of a neural network, converting its inputs into a probability distribution over the predicted output classes.

    Methods
    -------
    __call__(X: np.array) -> np.array:
        Applies the Softmax activation function to the input array.

    grad(X: np.array) -> int:
        Returns 1 as the gradient of the Softmax activation function. In the context of this class, the gradients are discarded.

    Examples
    --------
    >>> op = Softmax()
    >>> X = np.array([[1, 2, 3], [2, 3, 4]])
    >>> op(X)
    array([[0.09003057, 0.24472847, 0.66524096],
           [0.09003057, 0.24472847, 0.66524096]])
    >>> op.grad(X)
    1
    """

    def __call__(self, X: np.array) -> np.array:
        unique_values = dict(axis=1, keepdims=True)

        # subtracts the maximum value of each row of the input array, to prevent overflow of np.exp
        X_rel = X - X.max(**unique_values)

        exp_X_rel = np.exp(X_rel)
        return exp_X_rel / np.sum(exp_X_rel, axis=1, keepdims=True)

    def grad(self, X: np.array) -> int:
        return 1  # discard these gradients


class CrossEntropy(BaseLossFunc):
    """
    The Cross-Entropy loss function. It is commonly used in classification tasks. It measures the dissimilarity between the predicted probability distribution and the true distribution.

    Methods
    -------
    __call__(Y: np.array, Y_pred: np.array) -> float:
        Computes the Cross-Entropy loss between the true labels and the predicted labels.

    grad(Y: np.array, Y_pred: np.array) -> np.array:
        Computes the gradient of the Cross-Entropy loss with respect to the predicted labels.

    Examples
    --------
    >>> loss_func = CrossEntropy()
    >>> Y = np.array([[1, 0, 0], [0, 1, 0]])
    >>> Y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.5, 0.4]])
    >>> loss_func(Y, Y_pred)
    0.5249110451064818
    >>> loss_func.grad(Y, Y_pred)
    array([[-0.3,  0.2,  0.1],
           [ 0.1, -0.5,  0.4]])
    """

    def __call__(self, Y: np.array, Y_pred: np.array) -> float:
        epsilon = 1e-8
        return -np.sum(Y * np.log(Y_pred + epsilon)) / Y.shape[0]

    def grad(self, Y: np.array, Y_pred: np.array) -> np.array:
        return Y_pred - Y
