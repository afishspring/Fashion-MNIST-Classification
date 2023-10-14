import torch.nn as nn
import torch.nn.functional as F


class cnn_classify(nn.Module):
    def __init__(self, ) -> None:
        super(cnn_classify, self).__init__()

        self.conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.pooling = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 13 * 13, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pooling(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x
