from typing import List
from undergrad.ops import BaseOp
import numpy as np


class Model:
    def __init__(self, layers_dims: List[int],
                 activation_funcs: List[BaseOp],
                 initialization_method: str = "random"):
        self.layers_dims = layers_dims
        self.activation_funcs = activation_funcs
        self.weights, self.bias = self._initialize_model(initialization_method)

    def __len__(self):
        return len(self.weights)

    def __call__(self, X):
        return self.forward(X)

    def _initialize_model(self, method="random"):
        weights = []
        bias = []
        n_layers = len(self.layers_dims)
        for l in range(0, n_layers-1):
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

            weights.append(W.astype(np.float64))
            bias.append(b.astype(np.float64))

        return weights, bias

    def forward(self, X):
        activation = X
        self.activations = [X]
        self.Z_list = []

        for layer in range(len(self.weights)):
            z = np.dot(activation, self.weights[layer]) + self.bias[layer]
            self.Z_list.append(z)
            activation = self.activation_funcs[layer](z)
            self.activations.append(activation)
        return self.activations[-1]
