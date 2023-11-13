import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve
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


def plot_confusion_matrix(model, cm):
    plt.figure(figsize=(8, 8))
    sns.set(style="darkgrid")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('figure/confusion_matrix_'+model+'.jpg', dpi=1000)
    plt.show()
    
def plot_auroc(model, all_labels, all_predictions_pro):
    all_predictions_pro=np.vstack(all_predictions_pro)

    true_labels = np.array(all_labels)
    predicted_probabilities = np.array(all_predictions_pro)

    plt.figure(figsize=(10, 8))
    sns.set(style="darkgrid")
    for class_idx in range(predicted_probabilities.shape[1]):
        class_true_labels = true_labels == class_idx
        class_predicted_probs = predicted_probabilities[:, class_idx]
        fpr, tpr, _ = roc_curve(class_true_labels, class_predicted_probs)
        auroc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'Class {class_idx} (AUROC = {auroc:.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUROC Curve for Multi-Class Classification')
    plt.legend(loc="best")
    plt.savefig('figure/AUROC_'+model+'.jpg', dpi=1000)
    plt.show()


def plot_accuracy(model, accuracy):
    plt.figure()
    sns.set(style="darkgrid")
    plt.plot(accuracy, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Value Accuracy')
    plt.savefig('figure/value_accuracy_'+model+'.jpg', dpi=1000)
    plt.show()


def plot_loss(model, losses):
    plt.figure()
    sns.set(style="darkgrid")
    plt.plot(losses, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    plt.savefig('figure/train_loss_'+model+'.jpg', dpi=1000)
    plt.show()

def plot_compare_accuracy(model_configs, all_val_accuracies):
    plt.figure(figsize=(12, 6))
    sns.set(style="darkgrid")
    palette = sns.color_palette("husl", n_colors=len(model_configs))

    for i, model_name in enumerate(model_configs):
        color = palette[i]
        line_style = '--' if "without" in model_name else '-'
        plt.plot(all_val_accuracies[i], label=model_name+'_Accuracy', color=color, linestyle=line_style)

    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    # plt.title('Validation Accuracy for Different Kernel Size')
    plt.legend(loc='center right')
    plt.grid(True)
    plt.savefig('figure/value_accuracy_compare_kernel.jpg', dpi=1000)
    plt.show()

def plot_compare_loss(model_configs, all_train_losses):
    plt.figure(figsize=(12, 6))
    sns.set(style="darkgrid")
    palette = sns.color_palette("husl", n_colors=len(model_configs))
    for i, model_name in enumerate(model_configs):
        color = palette[i]
        line_style = '--' if "without" in model_name else '-'
        plt.plot(all_train_losses[i], label=model_name+'_Loss', color=color, linestyle=line_style)

    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    # plt.title('Training Loss for Different Kernel Size')
    plt.legend(loc='center right')
    plt.grid(True)
    plt.savefig('figure/train_loss_compare_kernel.jpg', dpi=1000)
    plt.show()