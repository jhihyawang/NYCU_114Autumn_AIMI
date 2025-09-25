import torch.nn as nn
import torchvision.models as models

def get_model(arch="resnet18", num_classes=2, pretrained=True):
    if arch == "resnet18":
        model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
    elif arch == "resnet50":
        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
    else:
        raise ValueError("Unsupported model arch")

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model