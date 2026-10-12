import math

import torch

_NUM_HEADS = 2
_HEAD_DIM = 8


@torch.fx.wrap
def multihead_attention(query, key, value, num_heads, head_dim):
    sequence_length, channels = query.size()
    query_heads = query.view(sequence_length, num_heads, head_dim).transpose(0, 1)
    key_heads = key.view(sequence_length, num_heads, head_dim).transpose(0, 1)
    value_heads = value.view(sequence_length, num_heads, head_dim).transpose(0, 1)
    scores = torch.matmul(
        query_heads,
        key_heads.transpose(-1, -2),
    ) * (1.0 / math.sqrt(head_dim))
    mask = torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
        device=query.device,
    ).tril()
    scores = scores.masked_fill(~mask, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    probabilities = torch.where(
        torch.isnan(probabilities),
        torch.zeros_like(probabilities),
        probabilities,
    )
    context = torch.matmul(probabilities, value_heads)
    return context.transpose(0, 1).reshape(sequence_length, channels)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads, head_dim):
        super().__init__()
        self.q_proj = torch.nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = torch.nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = torch.nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = torch.nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        context = multihead_attention(
            query,
            key,
            value,
            _NUM_HEADS,
            _HEAD_DIM,
        )
        return self.out_proj(context)


class GPT2Block(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads, head_dim):
        super().__init__()
        hidden_dim = 4 * embedding_dim
        self.ln_1 = torch.nn.LayerNorm(embedding_dim, eps=1e-5)
        self.attention = MultiHeadAttention(embedding_dim, num_heads, head_dim)
        self.ln_2 = torch.nn.LayerNorm(embedding_dim, eps=1e-5)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x):
        hidden = self.ln_1(x)
        hidden = self.attention(hidden)
        x = x + hidden
        residual = self.ln_2(x)
        residual = self.mlp(residual)
        return x + residual


def build_model(args):
    global _NUM_HEADS, _HEAD_DIM
    _NUM_HEADS = args.num_heads
    _HEAD_DIM = args.head_dim
    return GPT2Block(args.embedding_dim, args.num_heads, args.head_dim)


def example_input(args):
    return torch.randn(args.seq_len, args.embedding_dim, dtype=torch.float32)
