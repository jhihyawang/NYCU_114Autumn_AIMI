import torch
import torch.nn as nn


# Helper function to choose activation function
def get_activation(name="ELU", alpha=1.0):
    name = name.lower()
    if name == "elu":
        return nn.ELU(alpha)
    elif name == "relu":
        return nn.ReLU()
    elif name == "leakyrelu":
        return nn.LeakyReLU(0.1)
    else:
        print(f"[Warning] Unknown activation '{name}', defaulting to ELU.")
        return nn.ELU(alpha)


class EEGNet(nn.Module):
    def __init__(self, num_classes=2, Chans=2, Samples=750,
                 dropout_rate=0.25, activation="ELU", elu_alpha=1.0):
        super(EEGNet, self).__init__()

        act = get_activation(activation, elu_alpha)

        # ----- Block 1: Temporal Convolution -----
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1),
                      padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )

        # ----- Block 2: Depthwise Convolution (spatial filtering) -----
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1),
                      groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            act,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=dropout_rate)
        )

        # ----- Block 3: Separable Convolution (temporal filtering) -----
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1),
                      padding=(0, 7), groups=32, bias=False),  # depthwise
            nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False),  # pointwise
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            act,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=dropout_rate)
        )

        # ----- Compute flatten dimension dynamically -----
        with torch.no_grad():
            x = torch.zeros(1, 1, Chans, Samples)
            x = self.firstconv(x)
            x = self.depthwiseConv(x)
            x = self.separableConv(x)
            flatten_dim = x.view(1, -1).size(1)

        # ----- Classification layer -----
        self.classify = nn.Linear(flatten_dim, num_classes, bias=True)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(x.size(0), -1)
        return self.classify(x)
