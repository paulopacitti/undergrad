import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from undergrad import Model, Trainer
from undergrad.ops import ReLU, Softmax, CrossEntropy
from undergrad.optim import SGDOptimizer
from undergrad.metrics import plot_loss_history, accuracy_for_class


def setup_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    target_transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.LongTensor([x])),
        transforms.Lambda(lambda x: F.one_hot(x, 10)),
        transforms.Lambda(lambda x: x.squeeze(0))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True,
                                   transform=transform, target_transform=target_transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True,
                                  transform=transform, target_transform=target_transform)

    return train_dataset, test_dataset


def main():
    classes = [str(i) for i in range(10)]
    train_dataset, test_dataset = setup_dataset()
    train_dataset, validation_dataset = random_split(train_dataset, [0.8, 0.2])

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
    train_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = Model([784, 400, 100, 10], [ReLU(), ReLU(),
                  Softmax()], initialization_method="he")
    opt = SGDOptimizer(model, lr=1e-5)
    trainer = Trainer(model, opt, CrossEntropy())
    history = trainer.train(15, train_loader, validation_loader)
    plot_loss_history(history)
    accuracy_for_class(model, validation_loader, classes)


if __name__ == "__main__":
    main()