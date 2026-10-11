import torch


def build_model(args):
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            args.in_channels,
            args.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        torch.nn.ReLU(),
    )


def example_input(args):
    return torch.randn(
        args.batch,
        args.in_channels,
        args.height,
        args.width,
        dtype=torch.float32,
    )
