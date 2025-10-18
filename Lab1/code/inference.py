import warnings
from argparse import ArgumentParser
import os
import torch

from models import get_model
from dataloader import get_dataloaders
from utils import set_seed, test, plot_confusion_matrix

if __name__ == '__main__':
    set_seed(39)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    parser = ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--model', type=str, default='resnet18', help='Model name: resnet18, resnet50, vit_base_patch16_224')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='chest_xray')
    parser.add_argument('--resize', type=int, default=224)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # dataloader
    _, _, test_loader = get_dataloaders(
        data_dir=args.dataset,
        batch_size=args.batch_size,
        resize=args.resize
    )

    # load model
    model = get_model(args.model, args.num_classes)
    model_path = os.path.join("./weight", f"{args.model}_best.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # testing
    print(f"===== Testing Model [{args.model}] =====")
    test_acc, f1_score, best_c_matrix = test(test_loader, model, device)

    #Save confusion matrix
    folder = f'./result/{args.model}'
    os.makedirs(folder, exist_ok=True)
    plot_confusion_matrix(best_c_matrix, folder)