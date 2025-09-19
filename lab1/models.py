import torch.nn as nn
import torchvision.models as models

def get_model(model_name="resnet18", num_classes=2):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # 修改最後全連接層
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
