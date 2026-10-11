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
    },
    "normalization": {
        "batch_norm",
        "instance_norm",
        "layer_norm",
        "group_norm",
        "rms_norm",
        "local_response_norm",
        "fused_add_norm",
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

    print("operator kernels: PASS (supported subset)")


if __name__ == "__main__":
    main()
