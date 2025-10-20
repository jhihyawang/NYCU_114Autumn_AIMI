import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    """
    EEGNet implementation based on:
    Lawhern et al., 'EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces' (2018)
    (Version matched to screenshot reference)
    """

    def __init__(self, num_classes=2, Chans=2, Samples=750, dropout_rate=0.25):
        super(EEGNet, self).__init__()

        # Block 1: Temporal Convolution
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1),
                      padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )

        # Block 2: Depthwise Convolution (spatial filtering)
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1),
                      groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(alpha=1.0),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=dropout_rate)
        )

        # Block 3: Separable Convolution (temporal filtering)
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1),
                      padding=(0, 7), groups=32, bias=False),   # depthwise
            nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False),  # pointwise
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(alpha=1.0),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=dropout_rate)
        )

        # Classification layer (flatten dynamically)
        with torch.no_grad():
            x = torch.zeros(1, 1, Chans, Samples)
            x = self.firstconv(x)
            x = self.depthwiseConv(x)
            x = self.separableConv(x)
            flatten_dim = x.view(1, -1).size(1)

        self.classify = nn.Sequential(
            nn.Linear(flatten_dim, num_classes, bias=True)
        )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(x.size(0), -1)
        return self.classify(x)