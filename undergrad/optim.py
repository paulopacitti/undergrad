from abc import ABC, abstractmethod
from undergrad import Model
import numpy as np


class BaseOptimizer(ABC):
    def __init__(self, model: Model):
        self.model = model

    @abstractmethod
    def step(self, grads: np.array):
        pass


class SGDOptimizer(BaseOptimizer):
    def __init__(self, model: Model, lr=1e-3):
        super().__init__(model)
        self.lr = lr

    def step(self, grads: np.array):
        for i, (dW, db) in enumerate(grads):
            self.model.weights[i] -= self.lr * dW
            self.model.bias[i] -= self.lr * db


class AdamOptmizer(BaseOptimizer):
    def __init__(self, model: Model, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(model)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.v_dW = [np.zeros_like(self.model.weights[layer])
                     for layer in range(len(self.model.weights))]
        self.v_db = [np.zeros_like(self.model.bias[layer])
                     for layer in range(len(self.model.bias))]
        self.s_dW = [np.zeros_like(self.model.weights[layer])
                     for layer in range(len(self.model.weights))]
        self.s_db = [np.zeros_like(self.model.bias[layer])
                     for layer in range(len(self.model.bias))]
        self.iteration = 0

    def step(self, grads: np.array):
        self.iteration += 1
        for t, (dW, db) in enumerate(grads):
            # momentum and RMSprop
            self.v_dW[t] = self.beta1 * self.v_dW[t] + (1 - self.beta1) * dW
            self.s_dW[t] = self.beta2 * self.s_dW[t] + \
                (1 - self.beta2) * (dW ** 2)
            self.v_db[t] = self.beta1 * self.v_db[t] + (1 - self.beta1) * db
            self.s_db[t] = self.beta2 * self.s_db[t] + \
                (1 - self.beta2) * (db ** 2)

            #  correction
            v_correction_dW = self.v_dW[t] / \
                (1 - (self.beta1 ** (self.iteration)))
            s_correction_dW = self.s_dW[t] / \
                (1 - (self.beta2 ** (self.iteration)))
            v_correction_db = self.v_db[t] / \
                (1 - (self.beta1 ** (self.iteration)))
            s_correction_db = self.s_db[t] / \
                (1 - (self.beta2 ** (self.iteration)))

            #  update
            self.model.weights[t] -= self.lr * \
                (v_correction_dW / (np.sqrt(s_correction_dW) + self.epsilon))
            self.model.bias[t] -= self.lr * \
                (v_correction_db / (np.sqrt(s_correction_db) + self.epsilon))
