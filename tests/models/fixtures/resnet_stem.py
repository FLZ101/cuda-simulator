import torch


def build_model(args):
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            args.in_channels,
            args.out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        ),
        torch.nn.BatchNorm2d(args.out_channels),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )


def example_input(args):
    return torch.randn(
        args.batch,
        args.in_channels,
        args.height,
        args.width,
        dtype=torch.float32,
    )
