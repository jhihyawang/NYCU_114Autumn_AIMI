import torch.nn as nn
import torch
import torch.nn.functional as F


class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 64), stride=(1, 1), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 32), stride=(1, 1), padding=(0, 16), bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(1, 16), stride=(1, 1), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        # We'll calculate the correct size dynamically
        self.fc1 = None  # Will be initialized dynamically
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.ReLU()(x)
        
        # Initialize fc1 dynamically based on actual size
        x_flat = x.view(x.size(0), -1)
        if self.fc1 is None:
            flattened_size = x_flat.size(1)
            self.fc1 = nn.Linear(flattened_size, 128)
            if x.is_cuda:
                self.fc1 = self.fc1.cuda()
        
        x = self.fc1(x_flat)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

# (Optional) implement DeepConvNet model
class DeepConvNet(nn.Module):
    def __init__(self, C=2, T=750, N=2):
        super(DeepConvNet, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(1, 25, (1, 5), bias=True)             # Temporal conv
        self.conv2 = nn.Conv2d(25, 25, (C, 1), bias=True)            # Spatial conv across channels
        self.bn1 = nn.BatchNorm2d(25, eps=1e-5, momentum=0.1)
        self.pool1 = nn.MaxPool2d((1, 2))
        self.drop1 = nn.Dropout(0.5)

        # Block 2
        self.conv3 = nn.Conv2d(25, 50, (1, 5), bias=True)
        self.bn2 = nn.BatchNorm2d(50, eps=1e-5, momentum=0.1)
        self.pool2 = nn.MaxPool2d((1, 2))
        self.drop2 = nn.Dropout(0.5)

        # Block 3
        self.conv4 = nn.Conv2d(50, 100, (1, 5), bias=True)
        self.bn3 = nn.BatchNorm2d(100, eps=1e-5, momentum=0.1)
        self.pool3 = nn.MaxPool2d((1, 2))
        self.drop3 = nn.Dropout(0.5)

        # Block 4
        self.conv5 = nn.Conv2d(100, 200, (1, 5), bias=True)
        self.bn4 = nn.BatchNorm2d(200, eps=1e-5, momentum=0.1)
        self.pool4 = nn.MaxPool2d((1, 2))
        self.drop4 = nn.Dropout(0.5)

        # 計算最後 flatten 後的維度（根據公式）
        # T: 750 → 746 → 373 → 369 → 184 → 180 → 90 → 86 → 43
        feature_dim = 200 * 1 * 43
        self.fc = nn.Linear(feature_dim, N)

    def forward(self, x):
        # x: (batch, 1, C, T)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv3(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.conv4(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool3(x)
        x = self.drop3(x)

        x = self.conv5(x)
        x = self.bn4(x)
        x = F.elu(x)
        x = self.pool4(x)
        x = self.drop4(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
