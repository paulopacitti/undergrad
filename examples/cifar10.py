import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms

from undergrad import Model, Trainer
from undergrad.ops import ReLU, Softmax, CrossEntropy
from undergrad.optim import SGDOptimizer
from undergrad.metrics import plot_loss_history
import numpy as np


class CIFAR10(Dataset):
    def __init__(self, x, y=None, transform=None):
        self._x = x
        self._y = y  # .squeeze() if y is not None else None
        self._transform = transform

    def __len__(self):
        return len(self._x)

    def __getitem__(self, idx):
        image = self._x[idx]
        if self._transform is not None:
            image = self._transform(image)

        image = image.flatten()
        if self._y is None:
            return image

        # one hot encoding
        label = [0] * 10
        label[self._y[idx]] = 1
        return image, torch.Tensor(label)


def normalize(X):
    return (X - X.mean())/(X.std() + 1e-8)


# defina o caminho do conjunto de dados corretamente
dataset_path = 'data/dataset.npy'
dataset = np.load(dataset_path, allow_pickle=True).item()

x_train, y_train = dataset['train_images'], dataset['train_labels']
x_val, y_val = dataset['val_images'], dataset['val_labels']
x_test, y_test = dataset['test_images'], dataset['test_labels']
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

train_set = CIFAR10(x_train, y_train, transform=normalize)
val_set = CIFAR10(x_val, y_val, transform=normalize)
test_set = CIFAR10(x_test, y_test, transform=normalize)

train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)
test_loader = DataLoader(test_set, batch_size=32)

model = Model([3072, 500, 100, 10], [ReLU(), ReLU(),
              Softmax()], initialization_method="he")
opt = SGDOptimizer(model, lr=1e-5)
trainer = Trainer(model, opt, CrossEntropy())
history = trainer.train(15, train_loader, val_loader)
plot_loss_history(history)
