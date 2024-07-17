from typing import List, Literal, Tuple
from abc import ABC, abstractmethod
from undergrad.ops import BaseOp
import numpy as np


class Layer(ABC):
    @abstractmethod
    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input):
        pass

    @abstractmethod
    def backward(self, output_gradient, learning_rate):
        pass


class Dense(Layer):
    """
    Represents a dense (fully connected) layer within a neural network.

    This layer implements the forward and backward passes of a fully connected neural network layer. It includes an
    activation function that can be applied to the output of the layer.

    Attributes
    ----------
    - input_dim : int
        The dimensionality of the input to this layer.
    - output_dim : int
        The dimensionality of the output of this layer.
    - activation_func : BaseOp
        The activation function to be applied to the output of the layer.
    - weights : np.ndarray
        The weight matrix of the layer, initialized to zeros.
    - bias : np.ndarray
        The bias vector of the layer, initialized to zeros.
    - input : np.ndarray
        The last input to the layer. This is stored for use in the backward pass.
    - Z : np.ndarray
        The linear combination of inputs and weights, plus bias. This is stored for use in the backward pass.
    - activation : np.ndarray
        The output of the layer after applying the activation function.

    Methods
    -------
    - forward(X: np.array) -> np.array:
        Performs the forward pass of the layer using the input X. It computes the linear combination of inputs and
        weights, adds the bias, and then applies the activation function.

    - backward(output_gradient: np.array) -> np.array:
        Performs the backward pass of the layer. It computes the gradients of the loss with respect to the input of
        the layer, the weights, and the bias, using the gradient of the loss with respect to the output of the layer.

    Example
    -------
    >>> dense_layer = Dense(4, 2, ReLU())
    >>> X = np.array([[1, 2, 3, 4]])
    >>> output = dense_layer.forward(X)
    >>> print(output)
    >>> gradient = np.array([[0.5, -0.5]])
    >>> input_gradient, (dW, dB) = dense_layer.backward(gradient)
    """

    def __init__(self, input_dim: int, output_dim: int, activation_func: BaseOp):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_func = activation_func
        self.weights, self.bias = np.zeros((
            self.input_dim, self.output_dim)), np.zeros((1, self.output_dim))
        self.input = None
        self.Z = None
        self.activation = None

    def forward(self, X: np.array) -> np.array:
        """
        Performs the forward pass of the dense layer.

        This method computes the linear combination of the input data and the layer's weights, adds the bias, and then
        applies the activation function to produce the layer's output.

        Parameters
        ----------
        X : np.array
            The input data to the layer. It should have a shape that matches the expected input dimensionality of the layer.

        Returns
        -------
        np.array
            The output of the layer after applying the linear transformation and the activation function.

        Example
        -------
        >>> X = np.array([[1, 2, 3], [4, 5, 6]])
        >>> layer = Dense(input_dim=3, output_dim=2, activation_func=ReLU())
        >>> output = layer.forward(X)
        """

        self.input = X
        self.Z = np.dot(X, self.weights) + self.bias
        self.activation = self.activation_func(self.Z)
        return self.activation

    def backward(self, output_gradient: np.array) -> np.array:
        """
        Performs the backward pass of the dense layer.

        This method computes the gradients of the loss function with respect to the input of the layer, the weights, and the bias.
        It uses the gradient of the loss with respect to the output of this layer (output_gradient) to calculate these gradients.

        Parameters
        ----------
        - output_gradient : np.array
            The gradient of the loss function with respect to the output of this layer.

        Returns
        -------
        Tuple[np.array, Tuple[np.array, np.array]]
            - The gradient of the loss function with respect to the input of this layer.
            - A tuple of two arrays:
                - The gradient of the loss function with respect to the weights of this layer (dW).
                - The gradient of the loss function with respect to the bias of this layer (dB).

        Example
        -------
        >>> output_gradient = np.array([[0.5, -0.5]])
        >>> input_gradient, (dW, dB) = layer.backward(output_gradient)
        """
        activation_gradient = self.activation_func.grad(self.Z)

        # Compute the gradient of the input
        input_gradient = np.dot(
            output_gradient * activation_gradient, self.weights.T)

        dW = np.dot(self.input.T, output_gradient * activation_gradient)
        dB = np.sum(output_gradient * activation_gradient,
                    axis=0, keepdims=True)

        return input_gradient, (dW, dB)


class Model:
    """
    A class used to represent a neural network model.

    Parameters
    ----------
    - layers : List[Layer]
        A list of layers that make up the model.
    - initialization_method : Literal["random", "he", "xavier"]
        The method used to initialize the weights and biases of the model. It can be 'random', 'he', or 'xavier'.

    Methods
    -------
    - __len__():
        Returns the number of layers in the model.
    - __call__(X: np.array):
        Performs a forward pass through the model and returns the output.
    - _initialize_model(method: Literal["random", "he", "xavier"]):
        Initializes the weights and biases of the model.
    - forward(X: np.array):
        Performs a forward pass through the model and returns the output.

    Examples
    --------
    >>>  model = Model([Dense(3, 2, ReLU()), Dense(
    2, 3, ReLU()), Dense(3, 3, Softmax())], "xavier")
    >>> X = np.array([[1, 2, 3], [4, 5, 6]])
    >>> model(X)
    array([[0.47357281, 0.21650904, 0.30991815],
            [0.76226799, 0.04792249, 0.18980952]])
    """

    def __init__(self, layers: List[Layer], initialization_method: Literal["random", "he", "xavier"] = "random"):
        self.layers = layers
        self._initialize_layer(initialization_method)

    def __len__(self) -> int:
        """
        Returns the number of layers in the model.

        Returns
        -------
        int
            The number of layers in the model.

        Examples
        --------
        >>> model = Model([Dense(3, 2, ReLU()), Dense(
        2, 3, ReLU()), Dense(3, 3, Softmax())], "xavier")
        >>> len(model)
        3
        """

        return len(self.layers)

    def __call__(self, X: np.array) -> np.array:
        return self.forward(X)

    def _initialize_layer(self, method: str):
        """
        Initializes the weights and biases of each layer in the model according to the specified method.

        This method supports two initialization strategies: 'he' and 'xavier'. The 'he' initialization is designed for layers
        with ReLU activation functions, and it initializes the weights with values scaled according to the size of the previous
        layer, helping to maintain a controlled flow of gradients. The 'xavier' initialization is suitable for layers with
        sigmoid or tanh activation functions, scaling the weights to maintain the variance of activations across layers.

        Parameters
        ----------
        method : str
            The name of the initialization method to use. It can be either 'random' (default) 'he' or 'xavier'. The method name is case-insensitive.
        """
        for layer in self.layers:
            W = np.random.randn(layer.input_dim, layer.output_dim)
            b = np.random.randn(1, layer.output_dim)

            if method.lower() == 'he':
                W = W * np.sqrt(2/layer.input_dim)
                b = b * np.sqrt(2/layer.input_dim)

            if method.lower() == 'xavier':
                W = W * np.sqrt(1/np.mean([layer.input_dim, layer.output_dim]))
                b = b * np.sqrt(1/np.mean([layer.input_dim, layer.output_dim]))

            layer.weights, layer.bias = W.astype(
                np.float64), b.astype(np.float64)

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
        >>> model.forward(X)
        array([[0.47357281, 0.21650904, 0.30991815],
               [0.76226799, 0.04792249, 0.18980952]])
        """
        for layer in self.layers:
            X = layer.forward(X)

        return X
