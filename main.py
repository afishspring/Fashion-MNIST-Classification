import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from utils import load_data, plot_confusion_matrix, plot_accuracy, plot_loss
from models import cnn_classify

parser = argparse.ArgumentParser()
# 完整训练次数
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs to train.')
# 学习率
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
# batch_size
parser.add_argument('--batch_size', type=int, default=512,
                    help='batch_size')
# 权重衰减
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
args = parser.parse_args()

train_loader, val_loader, test_loader = load_data()

model = cnn_classify()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


def train():
    train_losses = []
    train_accuracy = []
    for epoch in range(args.epochs):
        losses = []
        accuracy = []
        loop = tqdm((train_loader), total=len(train_loader))
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            #前向 + 反向 + 更新
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            losses.append(loss)
            loss.backward()
            optimizer.step()

            _, predictions = outputs.max(1)
            num_correct = (predictions == labels).sum()
            running_train_acc = float(num_correct) / float(inputs.shape[0])
            accuracy.append(running_train_acc)

            loop.set_description(f'Epoch [{epoch}/{args.epochs}]')
            loop.set_postfix(loss=loss.item(), acc=running_train_acc)

        epoch_loss = (sum(losses) / len(losses)).item()
        epoch_accuracy = sum(accuracy) / len(accuracy)

        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
    
    print('finish train')
    plot_loss(train_losses)
    plot_accuracy(train_accuracy)

def test():
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    cm = confusion_matrix(all_labels, all_predictions)
    plot_confusion_matrix(cm)

    correct = sum(np.array(all_labels) == np.array(all_predictions))
    test_accuracy = correct / len(all_labels)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')


train()
test()