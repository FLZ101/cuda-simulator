#!/usr/bin/env python3

import argparse
import math
import os
import struct
import subprocess
import tempfile

# Operators currently executed in the real-GPU differential harness.
# Add new operators here and add a deterministic run_pair case below.
SUPPORTED_OPERATORS = {
    "bias_add",
    "batch_norm",
    "max_pool2d",
    "avg_pool2d",
    "layer_norm",
    "embedding_lookup",
    "transpose",
    "matmul_naive",
    "matmul_tiled",
    "softmax_row",
    "reduce_sum",
    "conv2d_naive",
    "conv2d_im2col",
    "conv2d_shared",
    "attention_scores",
    "gelu",
    "copy",
    "saxpy",
    "relu",
    "elementwise_add",
    "elementwise_mul",
    "scale",
    "mean",
    "dropout",
    "causal_mask",
    "residual_add",
    "rms_norm",
    "concat",
    "split",
    "argmax",
    "clamp",
    "neg",
    "square",
    "log_softmax",
    "cross_entropy",
    "tanh",
    "sigmoid",
    "silu",
    "reciprocal",
    "exp",
    "log",
    "sqrt",
    "pow",
    "rsqrt",
    "minimum",
    "maximum",
    "leaky_relu",
    "abs",
    "relu6",
    "softplus",
    "swiglu",
    "elu",
    "exp2",
    "log2",
    "floor",
    "ceil",
    "rint",
    "trunc",
    "acos",
    "erfc",
    "hard_sigmoid",
    "hardswish",
    "softsign",
    "log_sigmoid",
    "subtract",
    "divide",
    "copysign",
    "lerp",
    "cosh",
    "sinh",
    "asin",
    "atan",
    "cbrt",
    "mish",
    "expm1",
    "atan2",
    "hypot",
    "masked_fill",
    "alibi_bias",
    "global_avg_pool",
    "instance_norm",
    "pad2d",
    "upsample_nearest2d",
    "depthwise_conv2d",
    "repeat_kv",
    "channel_shuffle",
    "pixel_shuffle",
    "adaptive_avg_pool2d",
    "local_response_norm",
    "fused_add_norm",
    "cumsum",
    "group_norm",
    "conv_transpose2d",
    "pixel_unshuffle",
    "reflection_pad2d",
    "upsample_bilinear2d",
    "adaptive_max_pool2d",
    "rotary_embedding",
    "gather",
    "scatter_add",
    "tanhshrink",
    "selu",
    "gelu_tanh",
    "sign",
    "isfinite",
    "isinf",
    "isnan",
    "hardshrink",
    "softshrink",
    "prelu",
    "celu",
    "hardtanh",
    "threshold",
    "heaviside",
    "glu",
    "geglu",
    "reglu",
    "fmod",
    "remainder",
    "nextafter",
    "erf",
    "erfinv",
    "sinc",
    "lgamma",
    "digamma",
    "softmin",
    "softmax1d",
    "spatial_softmax",
    "reduce_prod",
    "reduce_min",
    "reduce_max",
    "argmin",
    "amax",
    "amin",
    "logsumexp",
    "norm",
    "std",
    "var",
    "cumprod",
    "cummax",
    "cummin",
    "count_nonzero",
    "all",
    "any",
    "nll_loss",
    "mse_loss",
    "l1_loss",
    "smooth_l1_loss",
    "huber_loss",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "kl_div",
    "poisson_nll_loss",
    "hinge_embedding_loss",
    "cosine_embedding_loss",
    "margin_ranking_loss",
    "triplet_margin_loss",
    "soft_margin_loss",
    "focal_loss",
    "cosine_similarity",
    "batch_matmul",
    "addmm",
    "outer_product",
    "dot_product",
    "einsum",
    "where",
    "select",
    "one_hot",
    "index_select",
    "index_copy",
    "index_fill",
    "masked_select",
    "take",
    "gather_nd",
    "scatter",
    "scatter_reduce",
    "batch_gather",
    "argsort",
    "sort",
    "topk",
    "unique",
    "nonzero",
    "stack",
    "tile",
    "repeat",
    "repeat_interleave",
    "expand",
    "narrow",
    "slice",
    "roll",
    "flip",
    "rot90",
    "masked_softmax",
    "scaled_dot_product_attention",
    "grouped_query_attention",
    "multi_query_attention",
    "multi_head_attention_forward",
    "flash_attention_simt",
    "sliding_window_attention",
    "block_sparse_attention",
    "relative_position_bias",
    "sinusoidal_position_embedding",
    "xpos_embedding",
    "ntk_rope_scaling",
    "yarn_rope_scaling",
    "kv_cache_append",
    "kv_cache_rotate",
    "kv_cache_update",
    "paged_kv_cache_read",
    "paged_kv_cache_write",
    "fused_mlp",
    "gated_mlp",
    "fused_rms_norm_linear",
    "fused_layer_norm_linear",
    "nms",
    "iou",
    "ciou",
    "roi_align",
    "box_decode",
    "box_encode",
    "rnn_cell",
    "lstm_cell",
    "gru_cell",
    "sequence_mask",
    "beam_search",
    "ctc_decode",
    "conv1d",
    "conv3d",
    "fold",
    "unfold",
    "deform_conv2d",
    "weight_norm",
    "spectral_norm",
    "median",
    "quantile",
    "mode",
    "searchsorted",
    "cross_attention",
    "local_attention",
    "permute",
    "reshape",
    "squeeze",
    "unsqueeze",
    "chunk",
    "unbind",
    "meshgrid",
    "swapaxes",
    "ctc_loss",
    "gaussian_nll_loss",
    "multilabel_margin_loss",
    "multimargin_loss",
    "tensordot",
    "kron",
    "mv",
    "bmm",
    "diou",
    "giou",
    "anchor_generate",
    "batched_nms",
    "generate_proposals",
    "ctc_prefix_beam_search",
    "attention_decoder",
    "transformer_decoder",
}

OPERATOR_GROUPS = {
    "elementwise": {
        "abs",
        "acos",
        "asin",
        "atan",
        "atan2",
        "cbrt",
        "ceil",
        "clamp",
        "copy",
        "copysign",
        "cosh",
        "divide",
        "elementwise_add",
        "elementwise_mul",
        "elu",
        "erfc",
        "exp",
        "exp2",
        "expm1",
        "floor",
        "gelu",
        "hard_sigmoid",
        "hardswish",
        "hypot",
        "leaky_relu",
        "lerp",
        "log",
        "log2",
        "log_sigmoid",
        "maximum",
        "minimum",
        "mish",
        "neg",
        "pow",
        "reciprocal",
        "relu",
        "relu6",
        "residual_add",
        "rint",
        "rsqrt",
        "saxpy",
        "scale",
        "sigmoid",
        "silu",
        "sinh",
        "softplus",
        "softsign",
        "sqrt",
        "square",
        "subtract",
        "swiglu",
        "tanh",
        "trunc",
        "tanhshrink",
        "selu",
        "gelu_tanh",
        "sign",
        "isfinite",
        "isinf",
        "isnan",
        "hardshrink",
        "softshrink",
        "prelu",
        "celu",
        "hardtanh",
        "threshold",
        "heaviside",
        "glu",
        "geglu",
        "reglu",
        "fmod",
        "remainder",
        "nextafter",
    },
    "convolution": {
        "avg_pool2d",
        "max_pool2d",
        "conv2d_naive",
        "conv2d_im2col",
        "conv2d_shared",
        "depthwise_conv2d",
        "conv_transpose2d",
        "adaptive_avg_pool2d",
        "adaptive_max_pool2d",
        "global_avg_pool",
        "pad2d",
        "reflection_pad2d",
        "upsample_nearest2d",
        "upsample_bilinear2d",
        "pixel_shuffle",
        "pixel_unshuffle",
        "channel_shuffle",
        "conv1d",
        "conv3d",
        "fold",
        "unfold",
        "deform_conv2d",
    },
    "normalization": {
        "batch_norm",
        "instance_norm",
        "layer_norm",
        "group_norm",
        "rms_norm",
        "local_response_norm",
        "fused_add_norm",
        "weight_norm",
        "spectral_norm",
    },
    "reduction": {
        "reduce_sum",
        "mean",
        "argmax",
        "cumsum",
        "reduce_prod",
        "reduce_min",
        "reduce_max",
        "argmin",
        "amax",
        "amin",
        "logsumexp",
        "norm",
        "std",
        "var",
        "cumprod",
        "cummax",
        "cummin",
        "count_nonzero",
        "all",
        "any",
        "median",
        "quantile",
        "mode",
        "searchsorted",
    },
    "attention": {
        "attention_scores",
        "softmax_row",
        "log_softmax",
        "causal_mask",
        "dropout",
        "masked_fill",
        "alibi_bias",
        "repeat_kv",
        "rotary_embedding",
        "embedding_lookup",
        "masked_softmax",
        "scaled_dot_product_attention",
        "grouped_query_attention",
        "multi_query_attention",
        "multi_head_attention_forward",
        "flash_attention_simt",
        "sliding_window_attention",
        "block_sparse_attention",
        "relative_position_bias",
        "sinusoidal_position_embedding",
        "xpos_embedding",
        "ntk_rope_scaling",
        "yarn_rope_scaling",
        "kv_cache_append",
        "kv_cache_rotate",
        "kv_cache_update",
        "paged_kv_cache_read",
        "paged_kv_cache_write",
        "fused_mlp",
        "gated_mlp",
        "fused_rms_norm_linear",
        "fused_layer_norm_linear",
        "cross_attention",
        "local_attention",
    },
    "indexing": {
        "concat",
        "split",
        "gather",
        "scatter_add",
        "where",
        "select",
        "one_hot",
        "index_select",
        "index_copy",
        "index_fill",
        "masked_select",
        "take",
        "gather_nd",
        "scatter",
        "scatter_reduce",
        "batch_gather",
        "argsort",
        "sort",
        "topk",
        "unique",
        "nonzero",
        "stack",
        "tile",
        "repeat",
        "repeat_interleave",
        "expand",
        "narrow",
        "slice",
        "roll",
        "flip",
        "rot90",
        "permute",
        "reshape",
        "squeeze",
        "unsqueeze",
        "chunk",
        "unbind",
        "meshgrid",
        "swapaxes",
    },
    "loss": {
        "cross_entropy",
        "nll_loss",
        "mse_loss",
        "l1_loss",
        "smooth_l1_loss",
        "huber_loss",
        "binary_cross_entropy",
        "binary_cross_entropy_with_logits",
        "kl_div",
        "poisson_nll_loss",
        "hinge_embedding_loss",
        "cosine_embedding_loss",
        "margin_ranking_loss",
        "triplet_margin_loss",
        "soft_margin_loss",
        "focal_loss",
        "cosine_similarity",
        "ctc_loss",
        "gaussian_nll_loss",
        "multilabel_margin_loss",
        "multimargin_loss",
    },
    "matrix": {
        "matmul_naive",
        "matmul_tiled",
        "transpose",
        "bias_add",
        "batch_matmul",
        "addmm",
        "outer_product",
        "dot_product",
        "einsum",
        "tensordot",
        "kron",
        "mv",
        "bmm",
    },
    "misc": {
        "erf",
        "erfinv",
        "sinc",
        "lgamma",
        "digamma",
        "softmin",
        "softmax1d",
        "spatial_softmax",
    },
    "detection": {
        "nms",
        "iou",
        "ciou",
        "roi_align",
        "box_decode",
        "box_encode",
        "diou",
        "giou",
        "anchor_generate",
        "batched_nms",
        "generate_proposals",
    },
    "sequence": {
        "rnn_cell",
        "lstm_cell",
        "gru_cell",
        "sequence_mask",
        "beam_search",
        "ctc_decode",
        "ctc_prefix_beam_search",
        "attention_decoder",
        "transformer_decoder",
    },
}


def write_floats(path, values):
    with open(path, "wb") as f:
        f.write(b"".join(struct.pack("<f", v) for v in values))


def write_ints(path, values):
    with open(path, "wb") as f:
        f.write(b"".join(struct.pack("<i", v) for v in values))


def read_floats(path):
    with open(path, "rb") as f:
        raw = f.read()
    return list(struct.unpack("<" + "f" * (len(raw) // 4), raw))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True)
    parser.add_argument("--cudart", required=True)
    parser.add_argument("--cuda", required=True)
    parser.add_argument("--core-dir", required=True)
    parser.add_argument("--group")
    args = parser.parse_args()

    selected = SUPPORTED_OPERATORS
    if args.group:
        if args.group not in OPERATOR_GROUPS:
            raise ValueError(f"unknown operator group: {args.group}")
        selected = OPERATOR_GROUPS[args.group]

    gpu_env = os.environ.copy()
    sim_env = os.environ.copy()
    sim_env["LD_LIBRARY_PATH"] = args.core_dir
    sim_env["LD_PRELOAD"] = f"{args.cudart}:{args.cuda}"

    with tempfile.TemporaryDirectory() as tmp:

        def run_pair(mode, make_args):
            if mode not in SUPPORTED_OPERATORS:
                print(f"{mode}: skip (not enabled yet)")
                return
            if mode not in selected:
                return
            gpu_out = os.path.join(tmp, f"{mode}_gpu.bin")
            sim_out = os.path.join(tmp, f"{mode}_sim.bin")
            subprocess.run([args.runner, mode, *make_args(gpu_out)], check=True, env=gpu_env)
            subprocess.run([args.runner, mode, *make_args(sim_out)], check=True, env=sim_env)
            gpu = read_floats(gpu_out)
            sim = read_floats(sim_out)
            assert len(gpu) == len(sim), f"{mode} length mismatch"
            for a, b in zip(gpu, sim):
                if abs(a - b) > 1e-4 + 1e-5 * abs(b):
                    raise AssertionError(f"{mode} value mismatch")

        def generic_args(
            a="-",
            b="-",
            c="-",
            d="-",
            idx="-",
            ints=(1, 0, 0, 0, 0, 0, 0, 0),
            floats=(0.0, 0.0, 0.0, 0.0),
        ):
            ints = list(ints) + [0] * (8 - len(ints))
            floats = list(floats) + [0.0] * (4 - len(floats))

            def make_args(out):
                return [a, b, c, d, idx, *[str(v) for v in ints], *[str(v) for v in floats], out]

            return make_args

        n = 128
        x = [float(i) for i in range(n)]
        x_path = os.path.join(tmp, "x.bin")
        write_floats(x_path, x)
        y = [float(i * 2) for i in range(n)]
        y_path = os.path.join(tmp, "y.bin")
        write_floats(y_path, y)
        signed = [float((i % 7) - 3) for i in range(n)]
        signed_path = os.path.join(tmp, "signed.bin")
        write_floats(signed_path, signed)
        unit = [float((i % 5) - 2) / 2.0 for i in range(n)]
        unit_path = os.path.join(tmp, "unit.bin")
        write_floats(unit_path, unit)
        pos = [float(i + 1) for i in range(n)]
        pos_path = os.path.join(tmp, "pos.bin")
        write_floats(pos_path, pos)
        special_values = [0.0, -0.0, 1.0, -1.0, math.inf, -math.inf, math.nan]
        special = [special_values[i % len(special_values)] for i in range(n)]
        special_path = os.path.join(tmp, "special.bin")
        write_floats(special_path, special)
        zero_mask = [1.0 if i % 2 == 0 else 0.0 for i in range(n)]
        zero_mask_path = os.path.join(tmp, "zero_mask.bin")
        write_floats(zero_mask_path, zero_mask)
        cnn_in = [float(i) for i in range(32)]
        cnn_in_path = os.path.join(tmp, "cnn_in.bin")
        write_floats(cnn_in_path, cnn_in)
        cnn_weight = [0.25] * 16
        cnn_weight_path = os.path.join(tmp, "cnn_weight.bin")
        write_floats(cnn_weight_path, cnn_weight)
        kv = [float((i % 5) - 2) for i in range(32)]
        kv_path = os.path.join(tmp, "kv.bin")
        write_floats(kv_path, kv)
        mask = [1.0 if i % 2 == 0 else 0.5 for i in range(n)]
        mask_path = os.path.join(tmp, "mask.bin")
        write_floats(mask_path, mask)
        causal = [float(i) for i in range(16)]
        causal_path = os.path.join(tmp, "causal.bin")
        write_floats(causal_path, causal)
        fused_residual = [float((i % 3) - 1) for i in range(16)]
        fused_residual_path = os.path.join(tmp, "fused_residual.bin")
        write_floats(fused_residual_path, fused_residual)
        cumsum_input = [float((i * 3) % 11) for i in range(16)]
        cumsum_path = os.path.join(tmp, "cumsum.bin")
        write_floats(cumsum_path, cumsum_input)
        conv_t_input = [float(i) for i in range(4)]
        conv_t_input_path = os.path.join(tmp, "conv_t_input.bin")
        write_floats(conv_t_input_path, conv_t_input)
        conv_t_weight = [0.25] * 4
        conv_t_weight_path = os.path.join(tmp, "conv_t_weight.bin")
        write_floats(conv_t_weight_path, conv_t_weight)
        rotary_input = [float((i % 7) - 3) for i in range(32)]
        rotary_input_path = os.path.join(tmp, "rotary_input.bin")
        write_floats(rotary_input_path, rotary_input)
        rotary_cos = [math.cos((i % 9) * 0.1) for i in range(16)]
        rotary_cos_path = os.path.join(tmp, "rotary_cos.bin")
        write_floats(rotary_cos_path, rotary_cos)
        rotary_sin = [math.sin((i % 9) * 0.1) for i in range(16)]
        rotary_sin_path = os.path.join(tmp, "rotary_sin.bin")
        write_floats(rotary_sin_path, rotary_sin)
        gather_indices = [2, 0, 3, 1]
        gather_indices_path = os.path.join(tmp, "gather_indices.bin")
        write_ints(gather_indices_path, gather_indices)
        scatter_indices = [0, 2, 0, 3]
        scatter_indices_path = os.path.join(tmp, "scatter_indices.bin")
        write_ints(scatter_indices_path, scatter_indices)
        targets = [0, 1]
        targets_path = os.path.join(tmp, "targets.bin")
        write_ints(targets_path, targets)
        run_pair("gelu", lambda out: [x_path, str(n), out])
        run_pair("copy", lambda out: [x_path, str(n), out])
        run_pair("relu", lambda out: [x_path, str(n), out])
        run_pair(
            "saxpy",
            lambda out: [x_path, y_path, "0.5", str(n), out],
        )
        run_pair(
            "elementwise_add",
            lambda out: [x_path, y_path, str(n), out],
        )
        run_pair(
            "elementwise_mul",
            lambda out: [x_path, y_path, str(n), out],
        )
        run_pair("scale", lambda out: [x_path, "0.5", str(n), out])
        run_pair("mean", lambda out: [x_path, str(n), out])
        run_pair("dropout", lambda out: [x_path, mask_path, str(n), out])
        run_pair("causal_mask", lambda out: [causal_path, "4", out])
        run_pair(
            "residual_add",
            lambda out: [x_path, y_path, str(n), out],
        )
        run_pair(
            "concat",
            lambda out: [x_path, y_path, str(n), str(n), out],
        )
        run_pair("split", lambda out: [x_path, "64", str(n), out])
        run_pair("argmax", lambda out: [x_path, str(n), out])
        run_pair("clamp", lambda out: [x_path, "0.0", "10.0", str(n), out])
        run_pair("neg", lambda out: [x_path, str(n), out])
        run_pair("square", lambda out: [x_path, str(n), out])
        run_pair("sigmoid", lambda out: [x_path, str(n), out])
        run_pair("tanh", lambda out: [x_path, str(n), out])
        run_pair("silu", lambda out: [x_path, str(n), out])
        run_pair("reciprocal", lambda out: [x_path, str(n), out])
        run_pair("exp", lambda out: [x_path, str(n), out])
        run_pair("log", lambda out: [y_path, str(n), out])
        run_pair("sqrt", lambda out: [y_path, str(n), out])
        run_pair("pow", lambda out: [y_path, "2.0", str(n), out])
        run_pair("rsqrt", lambda out: [y_path, str(n), out])
        run_pair("minimum", lambda out: [x_path, y_path, str(n), out])
        run_pair("maximum", lambda out: [x_path, y_path, str(n), out])
        run_pair("leaky_relu", lambda out: [signed_path, "0.01", str(n), out])
        run_pair("abs", lambda out: [signed_path, str(n), out])
        run_pair("relu6", lambda out: [signed_path, str(n), out])
        run_pair("softplus", lambda out: [signed_path, str(n), out])
        run_pair("swiglu", lambda out: [x_path, y_path, str(n), out])
        run_pair("elu", lambda out: [signed_path, "1.0", str(n), out])
        run_pair("exp2", lambda out: [signed_path, str(n), out])
        run_pair("log2", lambda out: [y_path, str(n), out])
        run_pair("floor", lambda out: [signed_path, str(n), out])
        run_pair("ceil", lambda out: [signed_path, str(n), out])
        run_pair("rint", lambda out: [signed_path, str(n), out])
        run_pair("trunc", lambda out: [signed_path, str(n), out])
        run_pair("acos", lambda out: [unit_path, str(n), out])
        run_pair("erfc", lambda out: [signed_path, str(n), out])
        run_pair("hard_sigmoid", lambda out: [signed_path, str(n), out])
        run_pair("hardswish", lambda out: [signed_path, str(n), out])
        run_pair("softsign", lambda out: [signed_path, str(n), out])
        run_pair("log_sigmoid", lambda out: [signed_path, str(n), out])
        run_pair("subtract", lambda out: [x_path, y_path, str(n), out])
        run_pair("divide", lambda out: [x_path, pos_path, str(n), out])
        run_pair("copysign", lambda out: [signed_path, x_path, str(n), out])
        run_pair("lerp", lambda out: [x_path, y_path, str(n), out])
        run_pair("cosh", lambda out: [signed_path, str(n), out])
        run_pair("sinh", lambda out: [signed_path, str(n), out])
        run_pair("asin", lambda out: [unit_path, str(n), out])
        run_pair("atan", lambda out: [signed_path, str(n), out])
        run_pair("cbrt", lambda out: [signed_path, str(n), out])
        run_pair("mish", lambda out: [signed_path, str(n), out])
        run_pair("expm1", lambda out: [signed_path, str(n), out])
        run_pair("atan2", lambda out: [signed_path, pos_path, str(n), out])
        run_pair("hypot", lambda out: [x_path, y_path, str(n), out])
        run_pair("tanhshrink", lambda out: [signed_path, str(n), out])
        run_pair("selu", lambda out: [signed_path, str(n), out])
        run_pair("gelu_tanh", lambda out: [signed_path, str(n), out])
        run_pair("sign", lambda out: [signed_path, str(n), out])
        run_pair("isfinite", lambda out: [special_path, str(n), out])
        run_pair("isinf", lambda out: [special_path, str(n), out])
        run_pair("isnan", lambda out: [special_path, str(n), out])
        run_pair("hardshrink", lambda out: [signed_path, str(n), "0.5", out])
        run_pair("softshrink", lambda out: [signed_path, str(n), "0.5", out])
        run_pair("prelu", lambda out: [signed_path, str(n), "0.25", out])
        run_pair("celu", lambda out: [signed_path, str(n), "1.0", out])
        run_pair(
            "hardtanh",
            lambda out: [signed_path, str(n), "-1.0", "1.0", out],
        )
        run_pair(
            "threshold",
            lambda out: [signed_path, str(n), "0.0", "-1.0", out],
        )
        run_pair(
            "heaviside",
            lambda out: [signed_path, str(n), "0.0", "1.0", out],
        )
        run_pair("glu", lambda out: [x_path, signed_path, str(n), out])
        run_pair("geglu", lambda out: [x_path, signed_path, str(n), out])
        run_pair("reglu", lambda out: [x_path, signed_path, str(n), out])
        run_pair("fmod", lambda out: [x_path, pos_path, str(n), out])
        run_pair("remainder", lambda out: [x_path, pos_path, str(n), out])
        run_pair("nextafter", lambda out: [x_path, pos_path, str(n), out])
        run_pair(
            "masked_fill",
            lambda out: [x_path, zero_mask_path, "-1.0", str(n), out],
        )
        run_pair(
            "alibi_bias",
            lambda out: [x_path, "2", "8", "4", out],
        )
        run_pair(
            "global_avg_pool",
            lambda out: [cnn_in_path, "4", "8", out],
        )
        run_pair(
            "instance_norm",
            lambda out: [cnn_in_path, "4", "8", "1e-5", out],
        )
        run_pair(
            "pad2d",
            lambda out: [cnn_in_path, "4", "4", "2", "1", "1", out],
        )
        run_pair(
            "upsample_nearest2d",
            lambda out: [cnn_in_path, "4", "4", "2", "2", out],
        )
        run_pair(
            "depthwise_conv2d",
            lambda out: [cnn_in_path, cnn_weight_path, "4", "4", "2", "3", "1", "2", "2", out],
        )
        run_pair(
            "repeat_kv",
            lambda out: [kv_path, "1", "2", "4", "4", "4", out],
        )
        run_pair(
            "channel_shuffle",
            lambda out: [cnn_in_path, "4", "2", "8", out],
        )
        run_pair(
            "pixel_shuffle",
            lambda out: [cnn_in_path, "4", "2", "2", "2", out],
        )
        run_pair(
            "adaptive_avg_pool2d",
            lambda out: [cnn_in_path, "4", "4", "2", "2", "1", out],
        )
        run_pair(
            "local_response_norm",
            lambda out: [cnn_in_path, "4", "8", "3", "1e-4", "0.75", "1.0", out],
        )
        run_pair(
            "fused_add_norm",
            lambda out: [cnn_in_path, fused_residual_path, "2", "8", "1e-5", out],
        )
        run_pair(
            "cumsum",
            lambda out: [cumsum_path, "2", "8", out],
        )
        run_pair(
            "group_norm",
            lambda out: [cnn_in_path, "4", "8", "2", "1e-5", out],
        )
        run_pair(
            "conv_transpose2d",
            lambda out: [
                conv_t_input_path,
                conv_t_weight_path,
                "1",
                "2",
                "2",
                "1",
                "2",
                "2",
                "1",
                "0",
                "3",
                "3",
                out,
            ],
        )
        run_pair(
            "pixel_unshuffle",
            lambda out: [cnn_in_path, "4", "2", "2", "2", out],
        )
        run_pair(
            "reflection_pad2d",
            lambda out: [cnn_in_path, "4", "2", "2", "1", out],
        )
        run_pair(
            "upsample_bilinear2d",
            lambda out: [cnn_in_path, "4", "2", "2", "2", out],
        )
        run_pair(
            "adaptive_max_pool2d",
            lambda out: [cnn_in_path, "4", "4", "2", "2", "1", out],
        )
        run_pair(
            "rotary_embedding",
            lambda out: [
                rotary_input_path,
                rotary_cos_path,
                rotary_sin_path,
                "4",
                "8",
                out,
            ],
        )
        run_pair(
            "gather",
            lambda out: [cnn_in_path, gather_indices_path, "4", "8", out],
        )
        run_pair(
            "scatter_add",
            lambda out: [cnn_in_path, scatter_indices_path, "4", "8", "4", out],
        )
        bias = [0.1, -0.2, 0.3, 0.4]
        bias_path = os.path.join(tmp, "bias.bin")
        write_floats(bias_path, bias)
        run_pair("bias_add", lambda out: [x_path, bias_path, str(n), str(4), out])

        scale = [1.0, 0.5, 2.0, 1.5]
        scale_path = os.path.join(tmp, "scale.bin")
        write_floats(scale_path, scale)
        run_pair(
            "batch_norm",
            lambda out: [x_path, scale_path, bias_path, str(4), str(32), "1e-5", out],
        )

        h = w = 8
        pool_input = [float(i) for i in range(h * w)]
        pool_path = os.path.join(tmp, "pool.bin")
        write_floats(pool_path, pool_input)
        run_pair(
            "max_pool2d",
            lambda out: [pool_path, str(h), str(w), str(1), str(4), str(2), out],
        )
        run_pair(
            "avg_pool2d",
            lambda out: [pool_path, str(h), str(w), str(1), str(4), str(2), out],
        )

        rows, hidden = 2, 16
        ln_input = [float((i % 7) - 3) for i in range(rows * hidden)]
        ln_path = os.path.join(tmp, "ln.bin")
        write_floats(ln_path, ln_input)
        run_pair("layer_norm", lambda out: [ln_path, str(rows), str(hidden), "1e-5", out])
        run_pair(
            "rms_norm",
            lambda out: [ln_path, str(rows), str(hidden), "1e-5", out],
        )

        vocab, dim = 8, 8
        weight = [float((i * 3) % 11) for i in range(vocab * dim)]
        weight_path = os.path.join(tmp, "weight.bin")
        write_floats(weight_path, weight)
        indices = [0, 2, 5, 7]
        indices_path = os.path.join(tmp, "indices.bin")
        write_ints(indices_path, indices)
        run_pair(
            "embedding_lookup",
            lambda out: [weight_path, indices_path, str(len(indices)), str(dim), out],
        )

        rows, cols = 4, 8
        trans_input = [float(i) for i in range(rows * cols)]
        trans_path = os.path.join(tmp, "trans.bin")
        write_floats(trans_path, trans_input)
        run_pair("transpose", lambda out: [trans_path, str(rows), str(cols), out])

        m, k, n = 8, 8, 8
        a = [float((i * 5) % 13) for i in range(m * k)]
        b = [float((i * 7) % 11) for i in range(k * n)]
        a_path = os.path.join(tmp, "a.bin")
        b_path = os.path.join(tmp, "b.bin")
        write_floats(a_path, a)
        write_floats(b_path, b)
        run_pair("matmul_naive", lambda out: [a_path, b_path, str(m), str(n), str(k), out])
        run_pair("matmul_tiled", lambda out: [a_path, b_path, str(m), str(n), str(k), out])

        rows, cols = 2, 8
        sm_input = [float((i % 5) - 2) for i in range(rows * cols)]
        sm_path = os.path.join(tmp, "sm.bin")
        write_floats(sm_path, sm_input)
        run_pair("softmax_row", lambda out: [sm_path, str(rows), str(cols), out])
        run_pair(
            "log_softmax",
            lambda out: [sm_path, str(rows), str(cols), out],
        )
        run_pair(
            "cross_entropy",
            lambda out: [sm_path, targets_path, str(rows), str(cols), out],
        )

        red_input = [float(i + 1) for i in range(256)]
        red_path = os.path.join(tmp, "red.bin")
        write_floats(red_path, red_input)
        run_pair("reduce_sum", lambda out: [red_path, str(256), str(256), out])

        c, h, w, out_c, kh, kw, out_h, out_w = 1, 4, 4, 1, 2, 2, 3, 3
        conv_in = [float(i) for i in range(c * h * w)]
        conv_w = [0.25] * (out_c * c * kh * kw)
        conv_in_path = os.path.join(tmp, "conv_in.bin")
        conv_w_path = os.path.join(tmp, "conv_w.bin")
        write_floats(conv_in_path, conv_in)
        write_floats(conv_w_path, conv_w)
        conv_args = [
            conv_in_path,
            conv_w_path,
            str(c),
            str(h),
            str(w),
            str(out_c),
            str(kh),
            str(kw),
            str(out_h),
            str(out_w),
        ]
        run_pair("conv2d_naive", lambda out: [*conv_args, out])
        run_pair("conv2d_im2col", lambda out: [*conv_args, out])
        run_pair("conv2d_shared", lambda out: [*conv_args, out])

        batch, heads, seq, dim = 1, 2, 4, 4
        q = [float((i % 5) - 2) for i in range(batch * heads * seq * dim)]
        attn_k = [float((i % 7) - 3) for i in range(batch * heads * seq * dim)]
        v = [float((i % 6) - 2) for i in range(batch * heads * seq * dim)]
        q_path = os.path.join(tmp, "q.bin")
        k_path = os.path.join(tmp, "k.bin")
        v_path = os.path.join(tmp, "v.bin")
        write_floats(q_path, q)
        write_floats(k_path, attn_k)
        write_floats(v_path, v)
        run_pair(
            "attention_scores",
            lambda out: [q_path, k_path, str(batch), str(heads), str(seq), str(dim), out],
        )

        attn_out_count = batch * heads * seq * dim
        run_pair(
            "masked_softmax",
            generic_args(sm_path, zero_mask_path, ints=(rows * cols, rows, cols)),
        )
        run_pair(
            "scaled_dot_product_attention",
            generic_args(q_path, k_path, v_path, ints=(attn_out_count, batch, heads, seq, dim)),
        )
        run_pair(
            "grouped_query_attention",
            generic_args(q_path, k_path, v_path, ints=(attn_out_count, batch, heads, seq, dim, 1)),
        )
        run_pair(
            "multi_query_attention",
            generic_args(q_path, k_path, v_path, ints=(attn_out_count, batch, heads, seq, dim)),
        )
        run_pair(
            "multi_head_attention_forward",
            generic_args(q_path, k_path, v_path, ints=(attn_out_count, batch, heads, seq, dim)),
        )
        run_pair(
            "flash_attention_simt",
            generic_args(q_path, k_path, v_path, ints=(attn_out_count, batch, heads, seq, dim)),
        )
        run_pair(
            "sliding_window_attention",
            generic_args(q_path, k_path, v_path, ints=(attn_out_count, batch, heads, seq, dim, 2)),
        )
        run_pair(
            "block_sparse_attention",
            generic_args(q_path, k_path, v_path, ints=(attn_out_count, batch, heads, seq, dim)),
        )
        run_pair(
            "relative_position_bias",
            generic_args(ints=(seq * seq, seq), floats=(1.0, 0.0, 0.0, 0.0)),
        )
        run_pair(
            "sinusoidal_position_embedding",
            generic_args(ints=(seq * 8, seq, 8)),
        )
        run_pair(
            "xpos_embedding",
            generic_args(ints=(seq * 8, seq, 8), floats=(0.5, 0.0, 0.0, 0.0)),
        )
        run_pair(
            "ntk_rope_scaling",
            generic_args(rotary_input_path, ints=(32, 16), floats=(0.5, 0.0, 0.0, 0.0)),
        )
        run_pair(
            "yarn_rope_scaling",
            generic_args(rotary_input_path, ints=(32, 16), floats=(0.5, 0.0, 0.0, 0.0)),
        )
        run_pair(
            "kv_cache_append",
            generic_args(kv_path, rotary_input_path, ints=(5 * 8, 4, 8)),
        )
        run_pair(
            "kv_cache_rotate",
            generic_args(kv_path, rotary_input_path, ints=(4 * 8, 4, 8)),
        )
        run_pair(
            "kv_cache_update",
            generic_args(
                kv_path, rotary_input_path, ints=(4 * 8, 4, 8), floats=(1.0, 0.0, 0.0, 0.0)
            ),
        )
        run_pair(
            "paged_kv_cache_read",
            generic_args(kv_path, "-", "-", "-", gather_indices_path, ints=(3 * 8, 4, 8, 3)),
        )
        run_pair(
            "paged_kv_cache_write",
            generic_args(x_path, kv_path, "-", "-", gather_indices_path, ints=(4 * 8, 4, 8, 3)),
        )

        mlp_m, mlp_k, mlp_h, mlp_n = 2, 4, 8, 4
        mlp_in = [float((i % 5) - 2) for i in range(mlp_m * mlp_k)]
        mlp_w1 = [float((i * 3) % 7) / 7.0 for i in range(mlp_k * mlp_h)]
        mlp_b1 = [0.05] * mlp_h
        mlp_w2 = [float((i * 5) % 6) / 6.0 for i in range(mlp_h * mlp_n)]
        mlp_in_path = os.path.join(tmp, "mlp_in.bin")
        mlp_w1_path = os.path.join(tmp, "mlp_w1.bin")
        mlp_b1_path = os.path.join(tmp, "mlp_b1.bin")
        mlp_w2_path = os.path.join(tmp, "mlp_w2.bin")
        write_floats(mlp_in_path, mlp_in)
        write_floats(mlp_w1_path, mlp_w1)
        write_floats(mlp_b1_path, mlp_b1)
        write_floats(mlp_w2_path, mlp_w2)
        run_pair(
            "fused_mlp",
            generic_args(
                mlp_in_path,
                mlp_w1_path,
                mlp_b1_path,
                mlp_w2_path,
                ints=(mlp_m * mlp_n, mlp_m, mlp_k, mlp_h, mlp_n),
            ),
        )
        run_pair(
            "gated_mlp",
            generic_args(
                mlp_in_path,
                mlp_w1_path,
                mlp_w1_path,
                mlp_w2_path,
                ints=(mlp_m * mlp_n, mlp_m, mlp_k, mlp_h, mlp_n),
            ),
        )
        norm_rows, norm_hidden, norm_out = 2, 16, 8
        norm_in = [float((i % 7) - 3) for i in range(norm_rows * norm_hidden)]
        norm_w = [float((i * 3) % 5) / 5.0 for i in range(norm_hidden * norm_out)]
        norm_gamma = [1.0 + (i % 3) * 0.25 for i in range(norm_hidden)]
        norm_bias = [0.1, -0.1, 0.2, 0.0, 0.05, -0.05, 0.1, 0.0]
        norm_in_path = os.path.join(tmp, "norm_in.bin")
        norm_w_path = os.path.join(tmp, "norm_w.bin")
        norm_gamma_path = os.path.join(tmp, "norm_gamma.bin")
        norm_bias_path = os.path.join(tmp, "norm_bias.bin")
        write_floats(norm_in_path, norm_in)
        write_floats(norm_w_path, norm_w)
        write_floats(norm_gamma_path, norm_gamma)
        write_floats(norm_bias_path, norm_bias)
        run_pair(
            "fused_rms_norm_linear",
            generic_args(
                norm_in_path,
                norm_w_path,
                norm_gamma_path,
                norm_bias_path,
                ints=(norm_rows * norm_out, norm_rows, norm_hidden, norm_out),
            ),
        )
        run_pair(
            "fused_layer_norm_linear",
            generic_args(
                norm_in_path,
                norm_w_path,
                norm_gamma_path,
                norm_bias_path,
                ints=(norm_rows * norm_out, norm_rows, norm_hidden, norm_out),
            ),
        )

        # Generic single-thread differential cases. The runner maps these
        # modes directly to tests/kernels/<group>/<mode>.cu.
        run_pair("erf", generic_args(unit_path, ints=(n, n)))
        run_pair("erfinv", generic_args(unit_path, ints=(n, n)))
        run_pair("sinc", generic_args(unit_path, ints=(n, n)))
        run_pair("lgamma", generic_args(pos_path, ints=(n, n)))
        run_pair("digamma", generic_args(pos_path, ints=(n, n)))
        run_pair("softmin", generic_args(signed_path, ints=(n, n)))
        run_pair("softmax1d", generic_args(signed_path, ints=(n, n)))
        sm_rows, sm_cols = 2, 8
        run_pair(
            "spatial_softmax",
            generic_args(sm_path, ints=(sm_rows * sm_cols, sm_rows, sm_cols)),
        )

        run_pair("reduce_prod", generic_args(unit_path, ints=(1, n)))
        run_pair("reduce_min", generic_args(signed_path, ints=(1, n)))
        run_pair("reduce_max", generic_args(signed_path, ints=(1, n)))
        run_pair("argmin", generic_args(signed_path, ints=(1, n)))
        run_pair("amax", generic_args(signed_path, ints=(1, n)))
        run_pair("amin", generic_args(signed_path, ints=(1, n)))
        run_pair("logsumexp", generic_args(signed_path, ints=(1, n)))
        run_pair("norm", generic_args(signed_path, ints=(1, n)))
        run_pair("std", generic_args(signed_path, ints=(1, n)))
        run_pair("var", generic_args(signed_path, ints=(1, n)))
        run_pair("count_nonzero", generic_args(signed_path, ints=(1, n)))
        run_pair("all", generic_args(unit_path, ints=(1, n)))
        run_pair("any", generic_args(signed_path, ints=(1, n)))
        cum_rows, cum_cols = 16, 8
        run_pair(
            "cumprod",
            generic_args(signed_path, ints=(cum_rows * cum_cols, cum_rows, cum_cols)),
        )
        run_pair(
            "cummax",
            generic_args(signed_path, ints=(cum_rows * cum_cols, cum_rows, cum_cols)),
        )
        run_pair(
            "cummin",
            generic_args(signed_path, ints=(cum_rows * cum_cols, cum_rows, cum_cols)),
        )

        prob = [0.01 + (i % 90) / 100.0 for i in range(n)]
        prob_path = os.path.join(tmp, "prob.bin")
        write_floats(prob_path, prob)
        run_pair("mse_loss", generic_args(x_path, y_path, ints=(1, n)))
        run_pair("l1_loss", generic_args(x_path, y_path, ints=(1, n)))
        run_pair(
            "smooth_l1_loss",
            generic_args(x_path, y_path, ints=(1, n), floats=(1.0, 0.0, 0.0, 0.0)),
        )
        run_pair(
            "huber_loss",
            generic_args(x_path, y_path, ints=(1, n), floats=(1.0, 0.0, 0.0, 0.0)),
        )
        run_pair(
            "binary_cross_entropy",
            generic_args(prob_path, zero_mask_path, ints=(1, n)),
        )
        run_pair(
            "binary_cross_entropy_with_logits",
            generic_args(signed_path, zero_mask_path, ints=(1, n)),
        )
        run_pair("kl_div", generic_args(prob_path, prob_path, ints=(1, n)))
        run_pair("poisson_nll_loss", generic_args(x_path, y_path, ints=(1, n)))
        run_pair("hinge_embedding_loss", generic_args(signed_path, unit_path, ints=(1, n)))
        cosine_targets = [1.0 if i % 2 == 0 else -1.0 for i in range(1)]
        cosine_targets_path = os.path.join(tmp, "cosine_targets.bin")
        write_floats(cosine_targets_path, cosine_targets)
        run_pair(
            "cosine_embedding_loss",
            generic_args(
                x_path, y_path, cosine_targets_path, ints=(1, n), floats=(0.5, 0.0, 0.0, 0.0)
            ),
        )
        ranking_targets = [1.0 if i % 2 == 0 else -1.0 for i in range(n)]
        ranking_targets_path = os.path.join(tmp, "ranking_targets.bin")
        write_floats(ranking_targets_path, ranking_targets)
        run_pair(
            "margin_ranking_loss",
            generic_args(
                x_path, y_path, ranking_targets_path, ints=(1, n), floats=(0.5, 0.0, 0.0, 0.0)
            ),
        )
        run_pair(
            "triplet_margin_loss",
            generic_args(x_path, y_path, signed_path, ints=(1, n), floats=(0.5, 0.0, 0.0, 0.0)),
        )
        run_pair("soft_margin_loss", generic_args(signed_path, zero_mask_path, ints=(1, n)))
        run_pair(
            "focal_loss",
            generic_args(prob_path, zero_mask_path, ints=(1, n), floats=(0.25, 2.0, 0.0, 0.0)),
        )
        run_pair("cosine_similarity", generic_args(x_path, y_path, ints=(1, n)))
        run_pair(
            "nll_loss",
            generic_args(sm_path, "-", "-", "-", targets_path, ints=(1, rows, cols)),
        )

        run_pair(
            "batch_matmul",
            generic_args(a_path, b_path, ints=(m * n, 1, m, k, n)),
        )
        mat_bias = [0.05] * (m * n)
        mat_bias_path = os.path.join(tmp, "mat_bias.bin")
        write_floats(mat_bias_path, mat_bias)
        run_pair(
            "addmm",
            generic_args(
                a_path, b_path, mat_bias_path, ints=(m * n, m, k, n), floats=(1.0, 1.0, 0.0, 0.0)
            ),
        )
        run_pair("outer_product", generic_args(x_path, y_path, ints=(8 * 8, 8, 8)))
        run_pair("dot_product", generic_args(x_path, y_path, ints=(1, 8)))
        run_pair("einsum", generic_args(a_path, b_path, ints=(m * n, m, k, n)))

        run_pair(
            "where",
            generic_args(zero_mask_path, x_path, y_path, ints=(n, n)),
        )
        run_pair(
            "select",
            generic_args(cnn_in_path, ints=(8, 4, 8, 2)),
        )
        run_pair(
            "one_hot",
            generic_args("-", "-", "-", "-", indices_path, ints=(4 * 8, 4, 8)),
        )
        run_pair(
            "index_select",
            generic_args(cnn_in_path, "-", "-", "-", gather_indices_path, ints=(4 * 8, 4, 8, 4)),
        )
        run_pair(
            "index_copy",
            generic_args(cnn_in_path, "-", x_path, "-", gather_indices_path, ints=(4 * 8, 4, 8, 4)),
        )
        run_pair(
            "index_fill",
            generic_args(
                cnn_in_path,
                "-",
                "-",
                "-",
                gather_indices_path,
                ints=(4 * 8, 4, 8, 4),
                floats=(7.0, 0.0, 0.0, 0.0),
            ),
        )
        run_pair(
            "masked_select",
            generic_args(x_path, zero_mask_path, ints=(n, n)),
        )
        run_pair(
            "take",
            generic_args(x_path, "-", "-", "-", indices_path, ints=(4, n, 4)),
        )
        run_pair(
            "gather_nd",
            generic_args(cnn_in_path, "-", "-", "-", indices_path, ints=(4, 32, 4)),
        )
        run_pair(
            "scatter",
            generic_args(cnn_in_path, "-", "-", "-", scatter_indices_path, ints=(4 * 8, 4, 8, 4)),
        )
        run_pair(
            "scatter_reduce",
            generic_args(cnn_in_path, "-", "-", "-", scatter_indices_path, ints=(4 * 8, 4, 8, 4)),
        )
        run_pair(
            "batch_gather",
            generic_args(x_path, "-", "-", "-", indices_path, ints=(4, 4, 32, 1)),
        )
        run_pair("argsort", generic_args(signed_path, ints=(n, n)))
        run_pair("sort", generic_args(signed_path, ints=(n, n)))
        run_pair("topk", generic_args(x_path, ints=(8, n, 8)))
        run_pair("unique", generic_args(signed_path, ints=(n, n)))
        run_pair("nonzero", generic_args(signed_path, ints=(n, n)))
        run_pair("stack", generic_args(x_path, y_path, ints=(2 * n, n)))
        run_pair("tile", generic_args(x_path, ints=(2 * n, n, 2)))
        run_pair("repeat", generic_args(x_path, ints=(2 * n, n, 2)))
        run_pair("repeat_interleave", generic_args(x_path, ints=(2 * n, n, 2)))
        run_pair("expand", generic_args(x_path, ints=(2 * n, n, 2)))
        run_pair("narrow", generic_args(x_path, ints=(32, 16, 32)))
        run_pair("slice", generic_args(x_path, ints=(32, 16, 32)))
        run_pair("roll", generic_args(x_path, ints=(n, n, 4)))
        run_pair("flip", generic_args(x_path, ints=(n, n)))
        run_pair("rot90", generic_args(cnn_in_path, ints=(4 * 8, 4, 8)))

        det_boxes1 = [
            0.0,
            0.0,
            4.0,
            4.0,
            2.0,
            2.0,
            6.0,
            6.0,
            5.0,
            1.0,
            8.0,
            5.0,
            1.0,
            1.0,
            3.0,
            3.0,
        ]
        det_boxes2 = [
            1.0,
            1.0,
            5.0,
            5.0,
            0.0,
            0.0,
            4.0,
            4.0,
            6.0,
            6.0,
            9.0,
            9.0,
        ]
        det_scores = [0.9, 0.7, 0.8, 0.6]
        det_boxes1_path = os.path.join(tmp, "det_boxes1.bin")
        det_boxes2_path = os.path.join(tmp, "det_boxes2.bin")
        det_scores_path = os.path.join(tmp, "det_scores.bin")
        write_floats(det_boxes1_path, det_boxes1)
        write_floats(det_boxes2_path, det_boxes2)
        write_floats(det_scores_path, det_scores)
        run_pair(
            "nms",
            generic_args(
                det_boxes1_path,
                det_scores_path,
                ints=(4, 4, 4),
                floats=(0.5, 0.0, 0.0, 0.0),
            ),
        )
        run_pair("iou", generic_args(det_boxes1_path, det_boxes2_path, ints=(12, 4, 3, 4)))
        run_pair("ciou", generic_args(det_boxes1_path, det_boxes2_path, ints=(12, 4, 3, 4)))

        det_feature = [float((i * 3) % 11) for i in range(4 * 4)]
        det_feature_path = os.path.join(tmp, "det_feature.bin")
        write_floats(det_feature_path, det_feature)
        det_rois = [
            0.0,
            0.0,
            1.0,
            1.0,
            0.25,
            0.25,
            0.75,
            0.75,
        ]
        det_rois_path = os.path.join(tmp, "det_rois.bin")
        write_floats(det_rois_path, det_rois)
        run_pair(
            "roi_align",
            generic_args(
                det_feature_path,
                det_rois_path,
                ints=(1 * 2 * 2 * 2, 1, 4, 4, 2, 2, 2),
            ),
        )

        det_anchors = [
            1.0,
            1.0,
            2.0,
            2.0,
            3.0,
            2.0,
            2.0,
            3.0,
            5.0,
            5.0,
            3.0,
            2.0,
            7.0,
            3.0,
            2.0,
            4.0,
        ]
        det_deltas = [
            0.0,
            0.0,
            0.1,
            -0.1,
            0.2,
            -0.2,
            0.0,
            0.1,
            -0.1,
            0.1,
            -0.2,
            0.0,
            0.0,
            0.0,
            0.1,
            0.1,
        ]
        det_anchors_path = os.path.join(tmp, "det_anchors.bin")
        det_deltas_path = os.path.join(tmp, "det_deltas.bin")
        write_floats(det_anchors_path, det_anchors)
        write_floats(det_deltas_path, det_deltas)
        run_pair(
            "box_decode",
            generic_args(det_anchors_path, det_deltas_path, ints=(16, 4, 4)),
        )
        det_gt = [
            0.0,
            0.0,
            3.0,
            4.0,
            2.0,
            1.0,
            6.0,
            7.0,
            4.0,
            4.0,
            9.0,
            8.0,
            6.0,
            2.0,
            10.0,
            9.0,
        ]
        det_gt_path = os.path.join(tmp, "det_gt.bin")
        write_floats(det_gt_path, det_gt)
        run_pair(
            "box_encode",
            generic_args(det_boxes1_path, det_gt_path, ints=(16, 4, 4)),
        )

        seq_batch, seq_input, seq_hidden = 2, 4, 3
        seq_x = [float((i % 5) - 2) for i in range(seq_batch * seq_input)]
        seq_w_ih = [float((i * 3) % 7) / 7.0 for i in range(seq_input * seq_hidden)]
        seq_w_hh = [float((i * 5) % 6) / 6.0 for i in range(seq_hidden * seq_hidden)]
        seq_prev_h = [float((i % 4) - 1) for i in range(seq_batch * seq_hidden)]
        seq_x_path = os.path.join(tmp, "seq_x.bin")
        seq_w_ih_path = os.path.join(tmp, "seq_w_ih.bin")
        seq_w_hh_path = os.path.join(tmp, "seq_w_hh.bin")
        seq_prev_h_path = os.path.join(tmp, "seq_prev_h.bin")
        write_floats(seq_x_path, seq_x)
        write_floats(seq_w_ih_path, seq_w_ih)
        write_floats(seq_w_hh_path, seq_w_hh)
        write_floats(seq_prev_h_path, seq_prev_h)
        run_pair(
            "rnn_cell",
            generic_args(
                seq_x_path,
                seq_w_ih_path,
                seq_w_hh_path,
                seq_prev_h_path,
                ints=(seq_batch * seq_hidden, seq_batch, seq_input, seq_hidden),
            ),
        )

        seq_w_ih_lstm = [float((i * 3) % 7) / 7.0 for i in range(4 * seq_hidden * seq_input)]
        seq_w_hh_lstm = [float((i * 5) % 6) / 6.0 for i in range(4 * seq_hidden * seq_hidden)]
        seq_prev_c = [0.05] * (seq_batch * seq_hidden)
        seq_w_ih_lstm_path = os.path.join(tmp, "seq_w_ih_lstm.bin")
        seq_w_hh_lstm_path = os.path.join(tmp, "seq_w_hh_lstm.bin")
        seq_states_path = os.path.join(tmp, "seq_states.bin")
        write_floats(seq_w_ih_lstm_path, seq_w_ih_lstm)
        write_floats(seq_w_hh_lstm_path, seq_w_hh_lstm)
        write_floats(seq_states_path, seq_prev_h + seq_prev_c)
        run_pair(
            "lstm_cell",
            generic_args(
                seq_x_path,
                seq_w_ih_lstm_path,
                seq_w_hh_lstm_path,
                seq_states_path,
                ints=(2 * seq_batch * seq_hidden, seq_batch, seq_input, seq_hidden),
            ),
        )

        seq_w_ih_gru = [float((i * 3) % 7) / 7.0 for i in range(3 * seq_hidden * seq_input)]
        seq_w_hh_gru = [float((i * 5) % 6) / 6.0 for i in range(3 * seq_hidden * seq_hidden)]
        seq_w_ih_gru_path = os.path.join(tmp, "seq_w_ih_gru.bin")
        seq_w_hh_gru_path = os.path.join(tmp, "seq_w_hh_gru.bin")
        write_floats(seq_w_ih_gru_path, seq_w_ih_gru)
        write_floats(seq_w_hh_gru_path, seq_w_hh_gru)
        run_pair(
            "gru_cell",
            generic_args(
                seq_x_path,
                seq_w_ih_gru_path,
                seq_w_hh_gru_path,
                seq_prev_h_path,
                ints=(seq_batch * seq_hidden, seq_batch, seq_input, seq_hidden),
            ),
        )

        seq_lengths = [3, 5]
        seq_lengths_path = os.path.join(tmp, "seq_lengths.bin")
        write_ints(seq_lengths_path, seq_lengths)
        run_pair(
            "sequence_mask",
            generic_args("-", "-", "-", "-", seq_lengths_path, ints=(12, 2, 6)),
        )

        beam_logits = [float((i * 7) % 13) for i in range(seq_batch * 8)]
        beam_logits_path = os.path.join(tmp, "beam_logits.bin")
        write_floats(beam_logits_path, beam_logits)
        run_pair(
            "beam_search",
            generic_args(beam_logits_path, ints=(6, 2, 8, 3)),
        )

        ctc_vocab, ctc_time, ctc_max = 8, 5, 4
        ctc_logits = [float((i * 5) % 11 - 5) for i in range(seq_batch * ctc_time * ctc_vocab)]
        ctc_logits_path = os.path.join(tmp, "ctc_logits.bin")
        write_floats(ctc_logits_path, ctc_logits)
        run_pair(
            "ctc_decode",
            generic_args(
                ctc_logits_path,
                "-",
                "-",
                "-",
                seq_lengths_path,
                ints=(
                    seq_batch * ctc_max,
                    seq_batch,
                    ctc_time,
                    ctc_vocab,
                    ctc_max,
                    0,
                    0,
                    ctc_vocab - 1,
                ),
            ),
        )

        conv1d_channels, conv1d_len, conv1d_out, conv1d_k = 2, 8, 3, 3
        conv1d_in = [float((i % 7) - 3) for i in range(conv1d_channels * conv1d_len)]
        conv1d_w = [
            float((i * 5) % 9) / 9.0 for i in range(conv1d_out * conv1d_channels * conv1d_k)
        ]
        conv1d_in_path = os.path.join(tmp, "conv1d_in.bin")
        conv1d_w_path = os.path.join(tmp, "conv1d_w.bin")
        write_floats(conv1d_in_path, conv1d_in)
        write_floats(conv1d_w_path, conv1d_w)
        run_pair(
            "conv1d",
            generic_args(
                conv1d_in_path,
                conv1d_w_path,
                ints=(
                    conv1d_out * 6,
                    conv1d_channels,
                    conv1d_len,
                    conv1d_out,
                    conv1d_k,
                    1,
                    0,
                    1,
                ),
            ),
        )

        conv3d_in = [float((i % 5) - 2) for i in range(4 * 4 * 4)]
        conv3d_w = [float((i * 3) % 7) / 7.0 for i in range(2 * 2 * 2)]
        conv3d_in_path = os.path.join(tmp, "conv3d_in.bin")
        conv3d_w_path = os.path.join(tmp, "conv3d_w.bin")
        write_floats(conv3d_in_path, conv3d_in)
        write_floats(conv3d_w_path, conv3d_w)
        run_pair(
            "conv3d",
            generic_args(conv3d_in_path, conv3d_w_path, ints=(27, 1, 4, 4, 4, 1, 2, 0)),
        )

        fold_in = [float((i % 4) - 1) for i in range(1 * 4 * 4)]
        fold_in_path = os.path.join(tmp, "fold_in.bin")
        write_floats(fold_in_path, fold_in)
        run_pair("fold", generic_args(fold_in_path, ints=(4, 1, 4, 2, 2, 2)))
        unfold_in = [float(i) for i in range(3 * 3)]
        unfold_in_path = os.path.join(tmp, "unfold_in.bin")
        write_floats(unfold_in_path, unfold_in)
        run_pair("unfold", generic_args(unfold_in_path, ints=(16, 1, 3, 3, 2)))

        deform_in = [float((i % 5) - 2) for i in range(4 * 4)]
        deform_w = [0.25] * 4
        deform_off = [0.0 if i % 2 == 0 else 0.1 for i in range(2 * 4 * 3 * 3)]
        deform_in_path = os.path.join(tmp, "deform_in.bin")
        deform_w_path = os.path.join(tmp, "deform_w.bin")
        deform_off_path = os.path.join(tmp, "deform_off.bin")
        write_floats(deform_in_path, deform_in)
        write_floats(deform_w_path, deform_w)
        write_floats(deform_off_path, deform_off)
        run_pair(
            "deform_conv2d",
            generic_args(
                deform_in_path,
                deform_w_path,
                deform_off_path,
                ints=(9, 4, 4, 2, 3, 3),
            ),
        )

        wn_rows, wn_cols = 2, 4
        wn_v = [float((i % 5) - 2) for i in range(wn_rows * wn_cols)]
        wn_g = [1.0, 2.0]
        wn_v_path = os.path.join(tmp, "wn_v.bin")
        wn_g_path = os.path.join(tmp, "wn_g.bin")
        write_floats(wn_v_path, wn_v)
        write_floats(wn_g_path, wn_g)
        run_pair(
            "weight_norm",
            generic_args(wn_v_path, wn_g_path, ints=(wn_rows * wn_cols, wn_rows, wn_cols)),
        )

        sn_rows, sn_cols = 3, 4
        sn_w = [float((i * 3) % 7) / 7.0 for i in range(sn_rows * sn_cols)]
        sn_v = [0.25] * sn_cols
        sn_w_path = os.path.join(tmp, "sn_w.bin")
        sn_v_path = os.path.join(tmp, "sn_v.bin")
        write_floats(sn_w_path, sn_w)
        write_floats(sn_v_path, sn_v)
        run_pair(
            "spectral_norm",
            generic_args(
                sn_w_path, "-", "-", sn_v_path, ints=(sn_rows * sn_cols, sn_rows, sn_cols)
            ),
        )

        red_median = [5.0, 3.0, 9.0, 3.0, 7.0, 3.0, 1.0, 8.0]
        red_median_path = os.path.join(tmp, "red_median.bin")
        write_floats(red_median_path, red_median)
        run_pair("median", generic_args(red_median_path, ints=(1, len(red_median))))
        run_pair(
            "quantile",
            generic_args(red_median_path, ints=(1, len(red_median)), floats=(0.5, 0.0, 0.0, 0.0)),
        )
        run_pair("mode", generic_args(red_median_path, ints=(1, len(red_median))))
        sorted_values = [1.0, 3.0, 5.0, 7.0, 9.0]
        sorted_values_path = os.path.join(tmp, "sorted_values.bin")
        write_floats(sorted_values_path, sorted_values)
        search_queries = [0.0, 3.0, 6.0, 10.0]
        search_queries_path = os.path.join(tmp, "search_queries.bin")
        write_floats(search_queries_path, search_queries)
        run_pair(
            "searchsorted",
            generic_args(sorted_values_path, search_queries_path, ints=(4, 5, 4)),
        )

        ca_batch, ca_q, ca_k, ca_dim = 2, 3, 4, 4
        ca_qv = [float((i % 5) - 2) for i in range(ca_batch * ca_q * ca_dim)]
        ca_kv = [float((i % 7) - 3) for i in range(ca_batch * ca_k * ca_dim)]
        ca_vv = [float((i % 6) - 2) for i in range(ca_batch * ca_k * ca_dim)]
        ca_q_path = os.path.join(tmp, "ca_q.bin")
        ca_k_path = os.path.join(tmp, "ca_k.bin")
        ca_v_path = os.path.join(tmp, "ca_v.bin")
        write_floats(ca_q_path, ca_qv)
        write_floats(ca_k_path, ca_kv)
        write_floats(ca_v_path, ca_vv)
        run_pair(
            "cross_attention",
            generic_args(
                ca_q_path,
                ca_k_path,
                ca_v_path,
                ints=(ca_batch * ca_q * ca_dim, ca_batch, ca_q, ca_k, ca_dim),
            ),
        )
        la_batch, la_heads, la_seq, la_dim = 1, 2, 4, 4
        la_q = [float((i % 5) - 2) for i in range(la_batch * la_heads * la_seq * la_dim)]
        la_k = [float((i % 7) - 3) for i in range(la_batch * la_heads * la_seq * la_dim)]
        la_v = [float((i % 6) - 2) for i in range(la_batch * la_heads * la_seq * la_dim)]
        la_q_path = os.path.join(tmp, "la_q.bin")
        la_k_path = os.path.join(tmp, "la_k.bin")
        la_v_path = os.path.join(tmp, "la_v.bin")
        write_floats(la_q_path, la_q)
        write_floats(la_k_path, la_k)
        write_floats(la_v_path, la_v)
        run_pair(
            "local_attention",
            generic_args(
                la_q_path,
                la_k_path,
                la_v_path,
                ints=(la_batch * la_heads * la_seq * la_dim, la_batch, la_heads, la_seq, la_dim, 2),
            ),
        )

        permute_in = [float(i) for i in range(2 * 3 * 4)]
        permute_in_path = os.path.join(tmp, "permute_in.bin")
        write_floats(permute_in_path, permute_in)
        run_pair("permute", generic_args(permute_in_path, ints=(24, 2, 3, 4, 1, 2, 0)))
        run_pair("reshape", generic_args(permute_in_path, ints=(24, 2, 3, 4, 4, 2, 3)))
        run_pair("squeeze", generic_args(x_path, ints=(8, 1, 8)))
        run_pair("unsqueeze", generic_args(x_path, ints=(8, 8)))
        chunk_in = [float(i) for i in range(10)]
        chunk_in_path = os.path.join(tmp, "chunk_in.bin")
        write_floats(chunk_in_path, chunk_in)
        run_pair("chunk", generic_args(chunk_in_path, ints=(10, 3, 10)))
        unbind_in = [float(i) for i in range(4 * 3)]
        unbind_in_path = os.path.join(tmp, "unbind_in.bin")
        write_floats(unbind_in_path, unbind_in)
        run_pair("unbind", generic_args(unbind_in_path, ints=(12, 4, 3)))
        mesh_x = [0.0, 1.0, 2.0]
        mesh_y = [10.0, 11.0, 12.0, 13.0]
        mesh_x_path = os.path.join(tmp, "mesh_x.bin")
        mesh_y_path = os.path.join(tmp, "mesh_y.bin")
        write_floats(mesh_x_path, mesh_x)
        write_floats(mesh_y_path, mesh_y)
        run_pair("meshgrid", generic_args(mesh_x_path, mesh_y_path, ints=(24, 3, 4)))
        swap_in = [float(i) for i in range(3 * 4)]
        swap_in_path = os.path.join(tmp, "swap_in.bin")
        write_floats(swap_in_path, swap_in)
        run_pair("swapaxes", generic_args(swap_in_path, ints=(12, 3, 4)))

        gaussian_var = [0.5 + (i % 3) * 0.25 for i in range(n)]
        gaussian_var_path = os.path.join(tmp, "gaussian_var.bin")
        write_floats(gaussian_var_path, gaussian_var)
        run_pair(
            "gaussian_nll_loss",
            generic_args(x_path, y_path, gaussian_var_path, ints=(1, n)),
        )
        loss_logits = [float((i % 7) - 3) for i in range(2 * 8)]
        loss_logits_path = os.path.join(tmp, "loss_logits.bin")
        write_floats(loss_logits_path, loss_logits)
        loss_targets = [0, 3]
        loss_targets_path = os.path.join(tmp, "loss_targets.bin")
        write_ints(loss_targets_path, loss_targets)
        run_pair(
            "multilabel_margin_loss",
            generic_args(
                loss_logits_path,
                "-",
                "-",
                "-",
                loss_targets_path,
                ints=(2, 2, 8),
            ),
        )
        run_pair(
            "multimargin_loss",
            generic_args(
                loss_logits_path,
                "-",
                "-",
                "-",
                loss_targets_path,
                ints=(2, 2, 8),
                floats=(1.0, 0.0, 0.0, 0.0),
            ),
        )
        run_pair(
            "ctc_loss",
            generic_args(
                ctc_logits_path,
                "-",
                "-",
                "-",
                seq_lengths_path,
                ints=(2, 2, 5, 8, 0, 0, 0, 7),
            ),
        )

        td_in = [float((i % 5) - 2) for i in range(2 * 3 * 4)]
        td_w = [float((i * 3) % 7) / 7.0 for i in range(4 * 5)]
        td_in_path = os.path.join(tmp, "td_in.bin")
        td_w_path = os.path.join(tmp, "td_w.bin")
        write_floats(td_in_path, td_in)
        write_floats(td_w_path, td_w)
        run_pair("tensordot", generic_args(td_in_path, td_w_path, ints=(30, 2, 3, 4, 5)))
        kron_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        kron_b = [0.5, -0.5, 1.0, 2.0]
        kron_a_path = os.path.join(tmp, "kron_a.bin")
        kron_b_path = os.path.join(tmp, "kron_b.bin")
        write_floats(kron_a_path, kron_a)
        write_floats(kron_b_path, kron_b)
        run_pair("kron", generic_args(kron_a_path, kron_b_path, ints=(24, 2, 3, 2, 2)))
        mv_a = [float((i % 4) - 1) for i in range(3 * 4)]
        mv_b = [0.25] * 4
        mv_a_path = os.path.join(tmp, "mv_a.bin")
        mv_b_path = os.path.join(tmp, "mv_b.bin")
        write_floats(mv_a_path, mv_a)
        write_floats(mv_b_path, mv_b)
        run_pair("mv", generic_args(mv_a_path, mv_b_path, ints=(3, 3, 4)))
        bmm_batch, bmm_m, bmm_k, bmm_n = 2, 3, 4, 2
        bmm_a = [float((i % 5) - 2) for i in range(bmm_batch * bmm_m * bmm_k)]
        bmm_b = [float((i * 3) % 7) / 7.0 for i in range(bmm_batch * bmm_k * bmm_n)]
        bmm_a_path = os.path.join(tmp, "bmm_a.bin")
        bmm_b_path = os.path.join(tmp, "bmm_b.bin")
        write_floats(bmm_a_path, bmm_a)
        write_floats(bmm_b_path, bmm_b)
        run_pair(
            "bmm",
            generic_args(
                bmm_a_path,
                bmm_b_path,
                ints=(bmm_batch * bmm_m * bmm_n, bmm_batch, bmm_m, bmm_k, bmm_n),
            ),
        )

        run_pair("diou", generic_args(det_boxes1_path, det_boxes2_path, ints=(12, 4, 3, 4)))
        run_pair("giou", generic_args(det_boxes1_path, det_boxes2_path, ints=(12, 4, 3, 4)))
        run_pair(
            "anchor_generate",
            generic_args(ints=(24, 2, 3, 4), floats=(1.0, 0.0, 0.0, 0.0)),
        )
        bnms_boxes = [
            0.0,
            0.0,
            4.0,
            4.0,
            2.0,
            2.0,
            6.0,
            6.0,
            5.0,
            1.0,
            8.0,
            5.0,
            0.0,
            0.0,
            3.0,
            3.0,
            1.0,
            1.0,
            5.0,
            5.0,
            6.0,
            6.0,
            9.0,
            9.0,
        ]
        bnms_scores = [0.9, 0.7, 0.8, 0.6, 0.95, 0.75]
        bnms_boxes_path = os.path.join(tmp, "bnms_boxes.bin")
        bnms_scores_path = os.path.join(tmp, "bnms_scores.bin")
        write_floats(bnms_boxes_path, bnms_boxes)
        write_floats(bnms_scores_path, bnms_scores)
        run_pair(
            "batched_nms",
            generic_args(
                bnms_boxes_path,
                bnms_scores_path,
                ints=(6, 2, 3),
                floats=(0.5, 0.0, 0.0, 0.0),
            ),
        )
        gp_scores = [0.6, 0.9, 0.4, 0.8, 0.5]
        gp_boxes = [
            0.0,
            0.0,
            2.0,
            2.0,
            1.0,
            1.0,
            3.0,
            3.0,
            2.0,
            2.0,
            4.0,
            4.0,
            3.0,
            3.0,
            5.0,
            5.0,
            4.0,
            4.0,
            6.0,
            6.0,
        ]
        gp_scores_path = os.path.join(tmp, "gp_scores.bin")
        gp_boxes_path = os.path.join(tmp, "gp_boxes.bin")
        write_floats(gp_scores_path, gp_scores)
        write_floats(gp_boxes_path, gp_boxes)
        run_pair("generate_proposals", generic_args(gp_scores_path, gp_boxes_path, ints=(12, 5, 3)))

        run_pair(
            "ctc_prefix_beam_search",
            generic_args(
                ctc_logits_path,
                "-",
                "-",
                "-",
                seq_lengths_path,
                ints=(6, 2, 5, 8, 3),
            ),
        )
        ad_enc = [float((i % 5) - 2) for i in range(2 * 4 * 4)]
        ad_query = [float((i % 7) - 3) for i in range(2 * 4)]
        ad_enc_path = os.path.join(tmp, "ad_enc.bin")
        ad_query_path = os.path.join(tmp, "ad_query.bin")
        write_floats(ad_enc_path, ad_enc)
        write_floats(ad_query_path, ad_query)
        run_pair(
            "attention_decoder",
            generic_args(ad_enc_path, ad_query_path, ints=(8, 2, 4, 4)),
        )
        tdec_in = [float((i % 5) - 2) for i in range(1 * 4 * 4)]
        tdec_in_path = os.path.join(tmp, "tdec_in.bin")
        write_floats(tdec_in_path, tdec_in)
        run_pair("transformer_decoder", generic_args(tdec_in_path, ints=(16, 1, 4, 4)))

    print("operator kernels: PASS (supported subset)")


if __name__ == "__main__":
    main()
