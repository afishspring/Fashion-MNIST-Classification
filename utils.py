import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_data():
    fashion_train_df = pd.read_csv('data/fashion-mnist_train.csv', sep=',')
    fashion_test_df = pd.read_csv('data/fashion-mnist_test.csv', sep=',')

    # convert to numpy arrays and reshape
    training = np.asarray(fashion_train_df, dtype='float32')
    X_train = training[:, 1:].reshape([-1, 1, 28, 28])
    X_train = X_train/255   # Normalizing the data
    y_train = training[:, 0]

    testing = np.asarray(fashion_test_df, dtype='float32')
    X_test = testing[:, 1:].reshape([-1, 1, 28, 28])
    X_test = X_test/255    # Normalizing the data
    y_test = testing[:, 0]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.3, random_state=2151137, shuffle=True)

    train = tensor_data(X_train, y_train)
    val = tensor_data(X_val, y_val)
    test = tensor_data(X_test, y_test)

    return train, val, test

# 使用分批训练数据


def tensor_data(X_data, y_data):
    torch_dataset = TensorDataset(
        torch.from_numpy(X_data), torch.from_numpy(y_data))
    loader = DataLoader(dataset=torch_dataset, batch_size=512, shuffle=True)
    return loader

# 绘制混淆矩阵


def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 8))
    sns.set()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def plot_accuracy(accuracy):
    plt.figure()
    plt.plot(accuracy, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training Accuracy')
    plt.show()


def plot_loss(losses):
    plt.figure()
    plt.plot(losses, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    plt.show()
