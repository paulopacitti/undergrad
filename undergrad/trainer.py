import numpy as np
from tqdm import tqdm


class Trainer:
    def __init__(self, model, optimizer, loss_func):
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.batch_size = 0

    def backward(self, Y):
        grads = []
        prediction = self.model.activations[-1]
        error = self.loss_func.grad(Y, prediction)

        for layer in range(len(self.model)-1, -1, -1):
            if layer == len(self.model)-1:
                delta = error * \
                    self.model.activation_funcs[layer].grad(
                        self.model.Z_list[layer])
            else:
                delta = np.dot(delta, self.model.weights[layer+1].T) * \
                    self.model.activation_funcs[layer].grad(
                        self.model.Z_list[layer])

            dW = np.dot(
                self.model.activations[layer].T, delta) / self.batch_size
            db = np.mean(delta, axis=0, keepdims=True)
            grads.append((dW, db))

        grads.reverse()
        return grads

    def train(self, n_epochs: int, train_loader, val_loader):
        log_dict = {'epoch': [],
                    'train_loss': [],
                    'val_loss': []}

        self.batch_size = train_loader.batch_size
        print("[training]:")
        for epoch in tqdm(range(n_epochs)):
            train_loss_history = []

            for i, batch in enumerate(train_loader):
                X, Y = batch
                X = X.numpy()
                Y = Y.numpy()
                Y_pred = self.model.forward(X)
                train_loss = self.loss_func(Y, Y_pred)
                train_loss_history.append(train_loss)

                grads = self.backward(Y)
                self.optimizer.step(grads)

            val_loss_history = []
            for i, batch in enumerate(val_loader):
                X, Y = batch
                X = X.numpy()
                Y = Y.numpy()
                Y_pred = self.model.forward(X)
                val_loss = self.loss_func(Y, Y_pred)
                val_loss_history.append(val_loss)

            # adding training loss to history
            train_loss = np.array(train_loss_history).mean()
            val_loss = np.array(val_loss_history).mean()

            log_dict['epoch'].append(epoch)
            log_dict['train_loss'].append(train_loss)
            log_dict['val_loss'].append(val_loss)

        return log_dict
