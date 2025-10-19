import torch.nn as nn
import torch


class DeepConvNet(nn.Module):
    def __init__(self, num_classes=2, channels=2, samples=750, dropout_rate=0.5):
        super(DeepConvNet, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(1, 25, (1, 10), bias=True)
        self.conv2 = nn.Conv2d(25, 25, (channels, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(25, eps=1e-5, momentum=0.1)
        self.pool1 = nn.MaxPool2d((1, 3))
        self.drop1 = nn.Dropout(dropout_rate)

        # Block 2
        self.conv3 = nn.Conv2d(25, 50, (1, 10), bias=True)
        self.bn2 = nn.BatchNorm2d(50, eps=1e-5, momentum=0.1)
        self.pool2 = nn.MaxPool2d((1, 3))
        self.drop2 = nn.Dropout(dropout_rate)

        # Block 3
        self.conv4 = nn.Conv2d(50, 100, (1, 10), bias=True)
        self.bn3 = nn.BatchNorm2d(100, eps=1e-5, momentum=0.1)
        self.pool3 = nn.MaxPool2d((1, 3))
        self.drop3 = nn.Dropout(dropout_rate)

        # Block 4
        self.conv5 = nn.Conv2d(100, 200, (1, 10), bias=True)
        self.bn4 = nn.BatchNorm2d(200, eps=1e-5, momentum=0.1)
        self.pool4 = nn.MaxPool2d((1, 3))
        self.drop4 = nn.Dropout(dropout_rate)

        # Calculate flattened dimension dynamically
        self.fc = None
        self.num_classes = num_classes

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.conv3(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Block 3
        x = self.conv4(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool3(x)
        x = self.drop3(x)

        # Block 4
        x = self.conv5(x)
        x = self.bn4(x)
        x = F.elu(x)
        x = self.pool4(x)
        x = self.drop4(x)

        # Flatten
        x_flat = x.view(x.size(0), -1)
        
        # Initialize fc dynamically
        if self.fc is None:
            flattened_size = x_flat.size(1)
            self.fc = nn.Linear(flattened_size, self.num_classes)
            if x.is_cuda:
                self.fc = self.fc.cuda()
        
        x = self.fc(x_flat)
        return x