import math

import torch


@torch.fx.wrap
def causal_softmax(scores, head_dim):
    size = scores.size(-1)
    mask = torch.ones(size, size, dtype=torch.bool, device=scores.device).tril()
    scaled = scores * (1.0 / math.sqrt(head_dim))
    masked = scaled.masked_fill(~mask, float("-inf"))
    probabilities = torch.softmax(masked, dim=-1)
    return torch.where(
        torch.isnan(probabilities),
        torch.zeros_like(probabilities),
        probabilities,
    )


class AttentionHead(torch.nn.Module):
    def __init__(self, embedding_dim, head_dim):
        super().__init__()
        self.q_proj = torch.nn.Linear(embedding_dim, head_dim, bias=False)
        self.k_proj = torch.nn.Linear(embedding_dim, head_dim, bias=False)
        self.v_proj = torch.nn.Linear(embedding_dim, head_dim, bias=False)
        self.out_proj = torch.nn.Linear(head_dim, embedding_dim, bias=False)

    def forward(self, x):
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        scores = torch.matmul(query, key.transpose(-1, -2))
        probabilities = causal_softmax(scores, query.size(-1))
        context = torch.matmul(probabilities, value)
        return self.out_proj(context)


def build_model(args):
    return AttentionHead(args.embedding_dim, args.head_dim)


def example_input(args):
    return torch.randn(args.seq_len, args.embedding_dim, dtype=torch.float32)
