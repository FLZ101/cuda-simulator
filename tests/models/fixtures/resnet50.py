import pathlib

import torch
from torchvision.models import ResNet50_Weights, resnet50

_WEIGHTS_PATH = pathlib.Path(__file__).resolve().parents[3] / ".cache" / "resnet50_imagenet_v2.pt"


def build_model(args):
    model = resnet50()
    if _WEIGHTS_PATH.is_file():
        state_dict = torch.load(_WEIGHTS_PATH, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    return model


def example_input(args):
    return torch.randn(
        args.batch,
        args.in_channels,
        args.height,
        args.width,
        dtype=torch.float32,
    )
