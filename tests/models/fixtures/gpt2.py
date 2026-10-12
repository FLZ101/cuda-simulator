import pathlib
import urllib.request

import torch

from models.fixtures import gpt2_block

_SEQ_LEN = 8
_PRETRAINED_URL = "https://huggingface.co/gpt2/resolve/main/pytorch_model.bin"
_PRETRAINED_FILE = ".cache/gpt2/pytorch_model.bin"


@torch.fx.wrap
def position_ids(token_ids, seq_len):
    return torch.arange(seq_len, device=token_ids.device, dtype=torch.long)


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


def _pretrained_checkpoint() -> pathlib.Path:
    checkpoint = _repo_root() / _PRETRAINED_FILE
    if not checkpoint.exists():
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_PRETRAINED_URL, checkpoint)
    return checkpoint


def _copy_(destination, source):
    if destination.shape != source.shape:
        raise RuntimeError(
            f"pretrained weight shape mismatch: {destination.shape} != {source.shape}"
        )
    destination.data.copy_(source)


def _load_pretrained(model, seq_len):
    checkpoint = torch.load(_pretrained_checkpoint(), map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)

    _copy_(model.wte.weight, state["wte.weight"])
    _copy_(model.wpe.weight, state["wpe.weight"][:seq_len])

    for index, block in enumerate(model.blocks):
        prefix = f"h.{index}."
        _copy_(block.ln_1.weight, state[prefix + "ln_1.weight"])
        _copy_(block.ln_1.bias, state[prefix + "ln_1.bias"])

        qkv = state[prefix + "attn.c_attn.weight"].t()
        query, key, value = qkv.chunk(3, dim=0)
        _copy_(block.attention.q_proj.weight, query)
        _copy_(block.attention.k_proj.weight, key)
        _copy_(block.attention.v_proj.weight, value)
        qkv_bias = state[prefix + "attn.c_attn.bias"]
        embedding_dim = block.attention.q_proj.out_features
        _copy_(block.attention.q_proj.bias, qkv_bias[:embedding_dim])
        _copy_(block.attention.k_proj.bias, qkv_bias[embedding_dim : 2 * embedding_dim])
        _copy_(block.attention.v_proj.bias, qkv_bias[2 * embedding_dim :])
        _copy_(block.attention.out_proj.weight, state[prefix + "attn.c_proj.weight"].t())
        _copy_(block.attention.out_proj.bias, state[prefix + "attn.c_proj.bias"])

        _copy_(block.ln_2.weight, state[prefix + "ln_2.weight"])
        _copy_(block.ln_2.bias, state[prefix + "ln_2.bias"])
        _copy_(block.mlp[0].weight, state[prefix + "mlp.c_fc.weight"].t())
        _copy_(block.mlp[0].bias, state[prefix + "mlp.c_fc.bias"])
        _copy_(block.mlp[2].weight, state[prefix + "mlp.c_proj.weight"].t())
        _copy_(block.mlp[2].bias, state[prefix + "mlp.c_proj.bias"])

    _copy_(model.ln_f.weight, state["ln_f.weight"])
    _copy_(model.ln_f.bias, state["ln_f.bias"])
    _copy_(model.lm_head.weight, state["wte.weight"])


class GPT2(torch.nn.Module):
    def __init__(self, vocab_size, n_layer, n_embd, n_head, head_dim, seq_len):
        super().__init__()
        self.wte = torch.nn.Embedding(vocab_size, n_embd)
        self.wpe = torch.nn.Embedding(seq_len, n_embd)
        self.blocks = torch.nn.ModuleList(
            [gpt2_block.GPT2Block(n_embd, n_head, head_dim) for _ in range(n_layer)]
        )
        self.ln_f = torch.nn.LayerNorm(n_embd, eps=1e-5)
        self.lm_head = torch.nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, token_ids):
        positions = position_ids(token_ids, _SEQ_LEN)
        hidden = self.wte(token_ids) + self.wpe(positions)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.ln_f(hidden)
        return self.lm_head(hidden)


def build_model(args):
    global _SEQ_LEN

    gpt2_block._NUM_HEADS = args.num_heads
    gpt2_block._HEAD_DIM = args.head_dim
    _SEQ_LEN = args.seq_len
    model = GPT2(
        args.vocab_size,
        args.n_layer,
        args.embedding_dim,
        args.num_heads,
        args.head_dim,
        args.seq_len,
    )
    if getattr(args, "pretrained", False):
        _load_pretrained(model, args.seq_len)
    return model


def example_input(args):
    return torch.arange(args.seq_len, dtype=torch.long) % args.vocab_size
