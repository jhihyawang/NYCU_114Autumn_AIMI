import torch
import torch.nn as nn

class DeepConvNet(nn.Module):
    def __init__(self, num_classes=2, channels=2, samples=750, dropout_rate=0.5):
        super(DeepConvNet, self).__init__()

        # ----- Block 1 -----
        self.conv1 = nn.Conv2d(1, 25, kernel_size=(1, 5), bias=False)           # temporal
        self.conv2 = nn.Conv2d(25, 25, kernel_size=(channels, 1), bias=False)   # spatial
        self.bn1   = nn.BatchNorm2d(25, eps=1e-5, momentum=0.1)
        self.elu1  = nn.ELU(alpha=1.0)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))
        self.drop1 = nn.Dropout(p=dropout_rate)

        # ----- Block 2 -----
        self.conv3 = nn.Conv2d(25, 50, kernel_size=(1, 5), bias=False)
        self.bn2   = nn.BatchNorm2d(50, eps=1e-5, momentum=0.1)
        self.elu2  = nn.ELU(alpha=1.0)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.drop2 = nn.Dropout(p=dropout_rate)

        # ----- Block 3 -----
        self.conv4 = nn.Conv2d(50, 100, kernel_size=(1, 5), bias=False)
        self.bn3   = nn.BatchNorm2d(100, eps=1e-5, momentum=0.1)
        self.elu3  = nn.ELU(alpha=1.0)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))
        self.drop3 = nn.Dropout(p=dropout_rate)

        # ----- Block 4 -----
        self.conv5 = nn.Conv2d(100, 200, kernel_size=(1, 5), bias=False)
        self.bn4   = nn.BatchNorm2d(200, eps=1e-5, momentum=0.1)
        self.elu4  = nn.ELU(alpha=1.0)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 2))
        self.drop4 = nn.Dropout(p=dropout_rate)

        # ----- Determine flattened dim (no helper method) -----
        with torch.no_grad():
            x = torch.zeros(1, 1, channels, samples)
            x = self.conv1(x); x = self.conv2(x); x = self.bn1(x); x = self.elu1(x); x = self.pool1(x); x = self.drop1(x)
            x = self.conv3(x); x = self.bn2(x); x = self.elu2(x); x = self.pool2(x); x = self.drop2(x)
            x = self.conv4(x); x = self.bn3(x); x = self.elu3(x); x = self.pool3(x); x = self.drop3(x)
            x = self.conv5(x); x = self.bn4(x); x = self.elu4(x); x = self.pool4(x); x = self.drop4(x)
            flatten_dim = x.view(1, -1).size(1)

        # ----- Classifier (no softmax; use CrossEntropyLoss) -----
        self.fc = nn.Linear(flatten_dim, num_classes)

    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.bn1(x); x = self.elu1(x); x = self.pool1(x); x = self.drop1(x)
        x = self.conv3(x); x = self.bn2(x); x = self.elu2(x); x = self.pool2(x); x = self.drop2(x)
        x = self.conv4(x); x = self.bn3(x); x = self.elu3(x); x = self.pool3(x); x = self.drop3(x)
        x = self.conv5(x); x = self.bn4(x); x = self.elu4(x); x = self.pool4(x); x = self.drop4(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)  # logits; CrossEntropyLoss handles LogSoftmax + NLLLoss
        return x