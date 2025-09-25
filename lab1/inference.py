import torch
from PIL import Image
from argparse import ArgumentParser

from models import get_model
from dataloader import get_transform


def predict(img_path, model, device):
    transform = get_transform(train=False)
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)
    return "NORMAL" if pred.item() == 0 else "PNEUMONIA"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--weights", type=str, default="resnet18_best.pth")
    parser.add_argument("--img", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(args.arch, num_classes=2, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    result = predict(args.img, model, device)
    print("Prediction:", result)