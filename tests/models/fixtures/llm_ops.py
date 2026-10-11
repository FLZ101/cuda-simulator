import torch


class Residual(torch.nn.Module):
    def __init__(self, features):
        super().__init__()
        self.norm = torch.nn.LayerNorm(features)
        self.linear = torch.nn.Linear(features, features)
        self.gelu = torch.nn.GELU()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        hidden = self.gelu(self.linear(self.norm(x)))
        return self.softmax((hidden + x) * x)


def build_model(args):
    return Residual(args.in_features)


def example_input(args):
    return torch.randn(args.batch, args.in_features, dtype=torch.float32)
