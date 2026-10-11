import torch


def build_model(args):
    return torch.nn.Sequential(
        torch.nn.Linear(args.in_features, args.hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden, args.out_features),
    )


def example_input(args):
    return torch.randn(args.batch, args.in_features, dtype=torch.float32)
