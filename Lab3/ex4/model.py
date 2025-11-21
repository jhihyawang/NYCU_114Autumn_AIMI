import torch.nn as nn
from torchvision import models

# --- Class names shared across your project ---
CLASS_NAMES = ["normal", "bacteria", "virus", "COVID-19"]

def build_model(model_name="resnet18", pretrained=True, num_classes=len(CLASS_NAMES)):
    """
    建立模型：
      支援 resnet18/50、efficientnet_b0/b1、convnext_tiny。
    """
    model_name = model_name.lower()

    # -------------------------------------------------
    # ConvNeXt Tiny
    # -------------------------------------------------
    if model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        return model
    
    if model_name == "convnext_small":
        weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        model = models.convnext_small(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        return model
    # -------------------------------------------------
    # ResNet 系列
    # -------------------------------------------------
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    
    elif model_name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    # -------------------------------------------------
    # EfficientNet 系列
    # -------------------------------------------------
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    elif model_name == "efficientnet_b1":
        weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b1(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    
    elif model_name == "efficientnet_b2":
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    
    elif model_name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    else:
        raise ValueError(
            f"目前僅支援 resnet18、resnet50、efficientnet_b0、efficientnet_b1、convnext_tiny，收到 {model_name}"
        )
