import numpy as np
import argparse
from tqdm import tqdm
import pickle
import os

import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils import *
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet50',
                    help='cnn/resnet34/.../resnet101')
# 完整训练次数
parser.add_argument('--epochs', type=int, default=100,
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
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def train(model):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                        lr=args.lr, weight_decay=args.weight_decay)

    train_losses = []
    val_accuracies = []
    
    for epoch in range(args.epochs):
        model.train()
        losses = []
        loop = tqdm((train_loader), total=len(train_loader))
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            loop.set_description(f'Train epoch [{epoch}/{args.epochs}]')
            loop.set_postfix(loss = loss.item())

        model.eval()
        accuracy = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predictions = outputs.max(1)
                num_correct = (predictions == labels).sum()
                running_train_acc = float(num_correct) / float(inputs.shape[0])
                accuracy.append(running_train_acc)
        
        train_loss = sum(losses)/len(losses)
        val_accurate = sum(accuracy)/len(accuracy)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %(epoch + 1, train_loss, val_accurate))
        
        train_losses.append(train_loss)
        val_accuracies.append(val_accurate)
    
    print('Finish train.')
    plot_loss(model_name, train_losses)
    plot_accuracy(model_name, val_accuracies)

    torch.save({'model': model.state_dict()}, 'pth/'+model_name+'.pth')
    return train_losses, val_accuracies

def test(model_name, using_exist=False):
    model = get_model(model_name)
    model.to(device)
    if using_exist:
        print("loading existed params")
        state_dict = torch.load('pth/'+model_name+'.pth')
        model.load_state_dict(state_dict['model'])
    else:
        train_losses, val_accuracies = train(model)

    all_labels = []
    all_predictions = []
    all_predictions_pro = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted_probabilities = torch.softmax(outputs, dim=1).cpu().detach().numpy()
            _, predictions = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_predictions_pro.extend(predicted_probabilities)
    
    test_accuracy = accuracy_score(all_labels, all_predictions)
    print(f'Accuracy: {test_accuracy:.3f}')

    report = classification_report(all_labels, all_predictions)
    print("Classification Report:")
    print(report)

    cm = confusion_matrix(all_labels, all_predictions)
    plot_confusion_matrix(model_name, cm)

    plot_auroc(model_name, all_labels, all_predictions_pro)

    return train_losses, val_accuracies

# model_configs = [
#     # 'resnet18_small',
#     'resnet18',
#     'resnet34',
#     'resnet50', 
#     'resnet101',
#     'resnet152',
#     'without_resnet18',
#     'without_resnet34',
#     'without_resnet50', 
#     'without_resnet101',
#     'without_resnet152',
#     # 'cnn'
# ]
model_configs = [
    'resnet18_small',
    'resnet18'
    # 'resnet34',
    # 'resnet50', 
    # 'resnet101',
    # 'resnet152',
    # 'without_resnet18',
    # 'without_resnet34',
    # 'without_resnet50', 
    # 'without_resnet101',
    # 'without_resnet152',
    # 'cnn'
]
all_train_losses = []
all_val_accuracies = []
for model_name in model_configs:
    print(f"Training {model_name}...")
    loss_acc_path = 'pkl/'+model_name+'.pkl'
    if os.path.exists(loss_acc_path):
        with open(loss_acc_path, 'rb') as f:
            data_set = pickle.load(f)
            train_losses = data_set['loss']
            val_accuracies = data_set['acc']
    else:
        train_losses, val_accuracies = test(model_name)
        data_set={
            'loss':train_losses,
            'acc':val_accuracies
        }
        with open(loss_acc_path, 'wb') as f:
            pickle.dump(data_set, f)
    
    all_train_losses.append(train_losses)
    all_val_accuracies.append(val_accuracies)

plot_compare_accuracy(model_configs, all_val_accuracies)
plot_compare_loss(model_configs, all_train_losses)