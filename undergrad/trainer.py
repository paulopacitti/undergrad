import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple

from undergrad import Model
from undergrad.ops import BaseLossFunc
from undergrad.optim import BaseOptimizer


class Trainer:
    """
    A class used to train a neural network model.

    The trainer takes a model, an optimizer, and a loss function, and provides methods for performing backpropagation 
    and training the model over multiple epochs.

    Parameters
    ----------
    model : Model
        The neural network model to be trained.
    optimizer : Optimizer
        The optimizer to be used to update the model's weights.
    loss_func : Loss
        The loss function to be used to compute the error between the model's predictions and the true values.

    Methods
    -------
    backward(Y: np.array):
        Performs backpropagation and returns the gradients of the weights and biases with respect to the loss.
    train(n_epochs: int, train_loader: DataLoader, validation_loader: DataLoader):
        Trains the model for a specified number of epochs and returns a log of the training and validation loss.

    Examples
    --------
    >>> model = Model([3, 2, 1], [ReLU(), Sigmoid()], "he")
    >>> optimizer = SGD(model, lr=0.01)
    >>> loss_func = CrossEntropyLoss()
    >>> trainer = Trainer(model, optimizer, loss_func)
    >>> train_loader = DataLoader(train_data, batch_size=32)
    >>> validation_loader = DataLoader(val_data, batch_size=32)
    >>> log = trainer.train(10, train_loader, validation_loader)
    """

    def __init__(self, model: Model, optimizer: BaseOptimizer, loss_func: BaseLossFunc) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.batch_size = 0

    def backward(self, Y: np.array) -> List[Tuple[np.array, np.array]]:
        """
        Performs backpropagation and returns the gradients of the weights and biases with respect to the loss.
        The gradients are computed by applying the chain rule to the computation graph of the model's forward pass.

        Parameters
        ----------
        Y : np.array
            The true values.

        Returns
        -------
        List[Tuple[np.array, np.array]]
            The gradients of the weights and biases with respect to the loss.
        """

        # initialize gradients
        grads = []
        # last layer activation
        prediction = self.model.activations[-1]
        # loss function gradient of the prediction
        error = self.loss_func.grad(Y, prediction)

        # backwards pass
        for layer in range(len(self.model)-1, -1, -1):
            # if last layer, use error when computing delta
            if layer == len(self.model)-1:
                delta = error * \
                    self.model.activation_funcs[layer].grad(
                        self.model.Z_list[layer])
            # otherwise use the delta from the previous layer
            else:
                delta = np.dot(delta, self.model.weights[layer+1].T) * \
                    self.model.activation_funcs[layer].grad(
                        self.model.Z_list[layer])

            # compute gradients of weights and biases
            dW = np.dot(
                self.model.activations[layer].T, delta) / self.batch_size
            db = np.mean(delta, axis=0, keepdims=True)
            # add to list of gradients
            grads.append((dW, db))

        # reverse list since the pass is backwards
        grads.reverse()
        return grads

    def train(self, n_epochs: int, train_loader: DataLoader, validation_loader: DataLoader) -> Dict[str, List[float]]:
        """
        Trains the model for a specified number of epochs and returns a log of the training and validation loss.

        The model is trained using mini-batch gradient descent. The training data is divided into batches using the DataLoader, and the model's weights are updated after each batch. The training and validation loss are computed after each epoch.

        Parameters
        ----------
        n_epochs : int
            The number of epochs to train the model for.
        train_loader : DataLoader
            The DataLoader for the training data.
        val_loader : DataLoader
            The DataLoader for the validation data.

        Returns
        -------
        Dict[str, List[float]]
            A log of the training and validation loss.
        """

        log_dict = {'epoch': [],
                    'train_loss': [],
                    'val_loss': []}
        self.batch_size = train_loader.batch_size

        print("[training]:")
        for epoch in tqdm(range(n_epochs)):
            # training loop
            train_loss_history = []
            for _, batch in enumerate(train_loader):
                # get input and labels from batch
                X, Y = batch
                # convert to numpy array, since undergrad works with numpy
                X = X.numpy()
                Y = Y.numpy()
                # forward pass to get predictions
                Y_pred = self.model.forward(X)
                # compute loss
                train_loss = self.loss_func(Y, Y_pred)
                train_loss_history.append(train_loss)

                # backpropagation to get grads
                grads = self.backward(Y)
                # optimizer step to update weights based on the grads
                self.optimizer.step(grads)

            # validation loop
            validation_loss_history = []
            for _, batch in enumerate(validation_loader):
                X, Y = batch
                X = X.numpy()
                Y = Y.numpy()
                Y_pred = self.model.forward(X)

                # compute loss but don't do backwards pass or do a optimizer step update weights,
                # since this is validation
                val_loss = self.loss_func(Y, Y_pred)
                validation_loss_history.append(val_loss)

            # adding training loss to history
            train_loss = np.array(train_loss_history).mean()
            val_loss = np.array(validation_loss_history).mean()

            log_dict['epoch'].append(epoch)
            log_dict['train_loss'].append(train_loss)
            log_dict['val_loss'].append(val_loss)

        return log_dict
