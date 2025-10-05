import torch.nn as nn
import torchvision.models as models
import timm

def get_model(name, num_classes):
    if name == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name.startswith('vit'):
        model = timm.create_model(name, pretrained=True, num_classes=num_classes)
    else:
        raise ValueError("Unsupported model arch")
    return model