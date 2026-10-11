import torch


def build_model(args):
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            args.in_channels,
            args.out_channels,
            kernel_size=3,
            padding=1,
        ),
        torch.nn.BatchNorm2d(args.out_channels),
        torch.nn.GELU(),
        torch.nn.AvgPool2d(kernel_size=2),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Dropout(0.5),
        torch.nn.Identity(),
    )


def example_input(args):
    return torch.randn(
        args.batch,
        args.in_channels,
        args.height,
        args.width,
        dtype=torch.float32,
    )
