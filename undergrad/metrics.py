import matplotlib.pyplot as plt
import numpy as np


def plot_loss_history(history):
    fig, ax = plt.subplots()
    ax.plot(history['train_loss'], color='#407cdb', label='Train')
    ax.plot(history['val_loss'], color='#db5740', label='Validation')

    ax.legend(loc='upper left')
    handles, labels = ax.get_legend_handles_labels()
    lgd = dict(zip(labels, handles))
    ax.legend(lgd.values(), lgd.keys())

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss x epoch')
    plt.show()


def accuracy_for_class(model, data_loader, classes):
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    for data in data_loader:
        X, Y = data
        X, Y = X.numpy(), Y.numpy()
        Y_pred = np.argmax(model.forward(X), axis=1)
        Y = np.argmax(Y, axis=1)

        # collect the correct predictions for each class
        for label, prediction in zip(Y, Y_pred):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.2f} %')
