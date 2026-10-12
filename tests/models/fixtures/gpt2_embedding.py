import torch


def build_model(args):
    return torch.nn.Sequential(
        torch.nn.Embedding(args.vocab_size, args.embedding_dim),
    )


def example_input(args):
    return torch.arange(args.seq_len, dtype=torch.long) % args.vocab_size
