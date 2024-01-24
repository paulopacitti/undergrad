from typing import List, Literal
from undergrad.ops import BaseOp
import numpy as np


class Model:
    """
    A class used to represent a neural network model.

    The model is defined by the dimensions of its layers, the activation functions used in each layer, and the method 
    used to initialize the weights and biases.

    Parameters
    ----------
    layers_dims : List[int]
        The dimensions of the layers in the model.
    activation_funcs : List[BaseOp]
        The activation functions to be used in each layer.
    initialization_method : str, optional
        The method to be used to initialize the weights and biases (default is "random").

    Methods
    -------
    __len__():
        Returns the number of layers in the model.
    __call__(X: np.array):
        Performs a forward pass through the model and returns the output.
    _initialize_model(method: Literal["random", "he", "xavier"]):
        Initializes the weights and biases of the model.
    forward(X: np.array):
        Performs a forward pass through the model and returns the output.

    Examples
    --------
    >>> model = Model([3, 2, 1], [ReLU(), Softmax()], "he")
    >>> X = np.array([[1, 2, 3], [4, 5, 6]])
    >>> model(X)
    array([[1.],
           [1.]])
    """

    def __init__(self, layers_dims: List[int],
                 activation_funcs: List[BaseOp],
                 initialization_method: Literal["random", "he", "xavier"] = "random"):
        self.layers_dims = layers_dims
        self.activation_funcs = activation_funcs
        self.weights, self.bias = self.__initialize_model(
            initialization_method)

    def __len__(self) -> int:
        """
        Returns the number of layers in the model.

        The number of layers is determined by the length of the weights list, as each element in the list corresponds 
        to the weights of one layer.

        Returns
        -------
        int
            The number of layers in the model.

        Examples
        --------
        >>> model = Model([3, 2, 1], [ReLU(), ReLU()], "he")
        >>> len(model)
        2
        """

        return len(self.weights)

    def __call__(self, X: np.array) -> np.array:
        return self.forward(X)

    def __initialize_model(self, method: str = "random") -> (List[np.array], List[np.array]):
        weights = []
        bias = []

        # initialize weights and biases for each layer except the input layer
        for l in range(0, len(self.layers_dims)-1):
            # initialize weights and biases with random values
            W = np.random.randn(self.layers_dims[l], self.layers_dims[l + 1])
            b = np.random.randn(1, self.layers_dims[l + 1])

            # He et al. initialization
            if method.lower() == 'he':
                W = W * np.sqrt(2/self.layers_dims[l])
                b = b * np.sqrt(2/self.layers_dims[l])

            # Xavier initialization
            if method.lower() == 'xavier':
                W = W * \
                    np.sqrt(
                        1/np.mean([self.layers_dims[l], self.layers_dims[l+1]]))
                b = b * \
                    np.sqrt(
                        1/np.mean([self.layers_dims[l], self.layers_dims[l+1]]))

            # append weights and biases to model current l layer
            weights.append(W.astype(np.float64))
            bias.append(b.astype(np.float64))

        return weights, bias

    def forward(self, X: np.array) -> np.array:
        """
        Performs a forward pass through the model and returns the output.

        The forward pass involves computing the dot product of the input data and the weights, adding the bias, and 
        applying the activation function for each layer. The activations and pre-activation values (Z) are stored for 
        each layer for use during backpropagation. Backprogation is performed by the Trainer class.

        Parameters
        ----------
        X : np.array
            The input data.

        Returns
        -------
        np.array
            The output of the model.

        Examples
        --------
        >>> model = Model([3, 2, 1], [ReLU(), Softmax()], "he")
        >>> X = np.array([[1, 2, 3], [4, 5, 6]])
        >>> model.forward(X)
        array([[1.],
               [1.]])
        """

        # input data for the first layer
        activation = X
        # activations for each layer, input data is considered the activation of the first layer
        self.activations = [X]
        # pre-activation values for each layer
        self.Z_list = []

        for layer in range(len(self.weights)):
            # compute pre-activation value
            z = np.dot(activation, self.weights[layer]) + self.bias[layer]
            self.Z_list.append(z)
            # apply activation function
            activation = self.activation_funcs[layer](z)
            self.activations.append(activation)

        # return output of the last layer
        return self.activations[-1]
