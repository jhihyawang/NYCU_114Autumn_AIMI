import torch
import torch.nn as nn
class DeepConvNet(nn.Module):
    def __init__(self, num_classes=2, Chans=2, Samples=750, dropout_rate=0.25):
        super(DeepConvNet, self).__init__()

        # ----- Block 1: Temporal + Spatial Convolutions -----
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), bias=False),        # temporal filtering
            nn.Conv2d(25, 25, kernel_size=(Chans, 1), bias=False),   # spatial filtering
            nn.BatchNorm2d(25, eps=1e-5, momentum=0.1),
            nn.ELU(alpha=1.0),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout_rate)
        )

        # ----- Block 2 -----
        self.block2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5), bias=False),
            nn.BatchNorm2d(50, eps=1e-5, momentum=0.1),
            nn.ELU(alpha=1.0),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout_rate)
        )

        # ----- Block 3 -----
        self.block3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), bias=False),
            nn.BatchNorm2d(100, eps=1e-5, momentum=0.1),
            nn.ELU(alpha=1.0),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout_rate)
        )

        # ----- Block 4 -----
        self.block4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), bias=False),
            nn.BatchNorm2d(200, eps=1e-5, momentum=0.1),
            nn.ELU(alpha=1.0),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout_rate)
        )

        # ----- Flatten dimension (calculated dynamically) -----
        with torch.no_grad():
            x = torch.zeros(1, 1, Chans, Samples)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            flatten_dim = x.view(1, -1).size(1)

        # ----- Classification Layer -----
        self.classify = nn.Linear(flatten_dim, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        return self.classify(x)