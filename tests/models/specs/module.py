#!/usr/bin/env python3

import argparse
import importlib.util
import operator
import pathlib

import torch
import torch.fx
import triton
import triton.language as tl

from ._common import compile_kernel, runner_prelude, write_floats


@triton.jit
def linear_kernel(x_ptr, w_ptr, b_ptr, out_ptr, m, n, k: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    acc = tl.load(b_ptr + col)
    for kk in tl.range(0, k):
        x = tl.load(x_ptr + row * k + kk)
        w = tl.load(w_ptr + col * k + kk)
        acc += x * w
    tl.store(out_ptr + row * n + col, acc)


@triton.jit
def relu_kernel(in_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, tl.maximum(x, 0.0), mask=mask)


@triton.jit
def conv2d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    S: tl.constexpr,
    stride: tl.constexpr,
    pad: tl.constexpr,
):
    pid = tl.program_id(0)
    oh = tl.program_id(1)
    ow = tl.program_id(2)
    n = pid // K
    k = pid % K
    acc = tl.load(b_ptr + k)
    out_h = (H + 2 * pad - R) // stride + 1
    out_w = (W + 2 * pad - S) // stride + 1
    for c in tl.range(0, C):
        for r in tl.range(0, R):
            for s in tl.range(0, S):
                ih = oh * stride - pad + r
                iw = ow * stride - pad + s
                in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                x = tl.load(
                    x_ptr + ((n * C + c) * H + ih) * W + iw,
                    mask=in_bounds,
                    other=0.0,
                )
                w = tl.load(w_ptr + ((k * C + c) * R + r) * S + s)
                acc += x * w
    tl.store(
        out_ptr + ((n * K + k) * out_h + oh) * out_w + ow,
        acc,
    )


@triton.jit
def batchnorm2d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    var_ptr,
    out_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    eps: tl.constexpr,
):
    pid = tl.program_id(0)
    hw = H * W
    c = (pid // hw) % C
    x = tl.load(x_ptr + pid)
    weight = tl.load(weight_ptr + c)
    bias = tl.load(bias_ptr + c)
    mean = tl.load(mean_ptr + c)
    var = tl.load(var_ptr + c)
    y = ((x - mean) / tl.sqrt(var + eps)) * weight + bias
    tl.store(out_ptr + pid, y)


@triton.jit
def maxpool2d_kernel(
    x_ptr,
    out_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    stride: tl.constexpr,
    pad: tl.constexpr,
):
    pid = tl.program_id(0)
    oh = tl.program_id(1)
    ow = tl.program_id(2)
    n = pid // C
    c = pid % C
    out_h = (H + 2 * pad - KH) // stride + 1
    out_w = (W + 2 * pad - KW) // stride + 1
    acc = float("-inf")
    for r in tl.range(0, KH):
        for s in tl.range(0, KW):
            ih = oh * stride - pad + r
            iw = ow * stride - pad + s
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            value = tl.load(
                x_ptr + ((n * C + c) * H + ih) * W + iw,
                mask=in_bounds,
                other=float("-inf"),
            )
            acc = tl.maximum(acc, value)
    tl.store(
        out_ptr + ((n * C + c) * out_h + oh) * out_w + ow,
        acc,
    )


@triton.jit
def copy_kernel(in_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def activation_kernel(
    in_ptr,
    out_ptr,
    n,
    BLOCK: tl.constexpr,
    mode: tl.constexpr,
    negative_slope: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(in_ptr + offsets, mask=mask)
    if mode == 0:
        y = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
    elif mode == 1:
        y = x * tl.sigmoid(x)
    elif mode == 2:
        y = 1.0 / (1.0 + tl.exp(-x))
    elif mode == 3:
        y = 2.0 * tl.sigmoid(2.0 * x) - 1.0
    elif mode == 4:
        y = tl.where(x >= 0, x, negative_slope * x)
    elif mode == 5:
        y = x * tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0) / 6.0
    elif mode == 6:
        y = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0) / 6.0
    elif mode == 7:
        y = tl.minimum(tl.maximum(x, 0.0), 6.0)
    elif mode == 8:
        y = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    elif mode == 9:
        y = tl.where(x > 0.0, x, tl.exp(x) - 1.0)
    elif mode == 10:
        y = 1.0507009873554805 * tl.where(x > 0.0, x, 1.6732632423543772 * (tl.exp(x) - 1.0))
    elif mode == 11:
        y = tl.maximum(x, 0.0) + tl.minimum(0.0, tl.exp(x) - 1.0)
    elif mode == 12:
        softplus = tl.log(1.0 + tl.exp(x))
        y = x * (2.0 * tl.sigmoid(2.0 * softplus) - 1.0)
    elif mode == 13:
        y = tl.minimum(tl.maximum(x, -1.0), 1.0)
    elif mode == 14:
        y = x / (1.0 + tl.abs(x))
    else:
        y = -tl.log(1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def avgpool2d_kernel(
    x_ptr,
    out_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    stride: tl.constexpr,
    pad: tl.constexpr,
):
    pid = tl.program_id(0)
    oh = tl.program_id(1)
    ow = tl.program_id(2)
    n = pid // C
    c = pid % C
    out_h = (H + 2 * pad - KH) // stride + 1
    out_w = (W + 2 * pad - KW) // stride + 1
    acc = 0.0
    for r in tl.range(0, KH):
        for s in tl.range(0, KW):
            ih = oh * stride - pad + r
            iw = ow * stride - pad + s
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            value = tl.load(
                x_ptr + ((n * C + c) * H + ih) * W + iw,
                mask=in_bounds,
                other=0.0,
            )
            acc += value
    tl.store(
        out_ptr + ((n * C + c) * out_h + oh) * out_w + ow,
        acc / (KH * KW),
    )


@triton.jit
def adaptive_avgpool2d_1x1_kernel(
    x_ptr,
    out_ptr,
    H: tl.constexpr,
    W: tl.constexpr,
):
    pid = tl.program_id(0)
    acc = 0.0
    for h in tl.range(0, H):
        for w in tl.range(0, W):
            acc += tl.load(x_ptr + pid * H * W + h * W + w)
    tl.store(out_ptr + pid, acc / (H * W))


@triton.jit
def adaptive_maxpool2d_1x1_kernel(
    x_ptr,
    out_ptr,
    H: tl.constexpr,
    W: tl.constexpr,
):
    pid = tl.program_id(0)
    acc = float("-inf")
    for h in tl.range(0, H):
        for w in tl.range(0, W):
            acc = tl.maximum(acc, tl.load(x_ptr + pid * H * W + h * W + w))
    tl.store(out_ptr + pid, acc)


@triton.jit
def binary_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr, mode: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    y = a + b if mode == 0 else a * b
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def concat_last_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n,
    BLOCK: tl.constexpr,
    A_INNER: tl.constexpr,
    B_INNER: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    inner = offsets % (A_INNER + B_INNER)
    outer = offsets // (A_INNER + B_INNER)
    mask = offsets < n
    a_mask = mask & (inner < A_INNER)
    b_mask = mask & (inner >= A_INNER)
    a = tl.load(a_ptr + outer * A_INNER + inner, mask=a_mask, other=0.0)
    b = tl.load(b_ptr + outer * B_INNER + (inner - A_INNER), mask=b_mask, other=0.0)
    tl.store(out_ptr + offsets, a + b, mask=mask)


@triton.jit
def transpose_kernel(
    x_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < M * N
    row = offsets // N
    col = offsets % N
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + col * M + row, x, mask=mask)


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    acc = 0.0
    for kk in tl.range(0, K):
        a = tl.load(a_ptr + row * K + kk)
        b = tl.load(b_ptr + kk * N + col)
        acc += a * b
    tl.store(out_ptr + row * N + col, acc)


@triton.jit
def layernorm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // N
    col = pid % N
    mean = 0.0
    for j in tl.range(0, N):
        mean += tl.load(x_ptr + row * N + j)
    mean = mean / N
    var = 0.0
    for j in tl.range(0, N):
        value = tl.load(x_ptr + row * N + j)
        var += (value - mean) * (value - mean)
    var = var / N
    x = tl.load(x_ptr + pid)
    weight = tl.load(weight_ptr + col)
    bias = tl.load(bias_ptr + col)
    y = (x - mean) / tl.sqrt(var + eps) * weight + bias
    tl.store(out_ptr + pid, y)


@triton.jit
def softmax_kernel(x_ptr, out_ptr, N: tl.constexpr, log: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // N
    col = pid % N
    max_value = float("-inf")
    for j in tl.range(0, N):
        max_value = tl.maximum(max_value, tl.load(x_ptr + row * N + j))
    denominator = 0.0
    for j in tl.range(0, N):
        denominator += tl.exp(tl.load(x_ptr + row * N + j) - max_value)
    x = tl.load(x_ptr + pid)
    y = x - max_value
    result = y - tl.log(denominator) if log else tl.exp(y) / denominator
    tl.store(out_ptr + pid, result)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--module-file", required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--in-features", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--out-features", type=int, default=8)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--out-channels", type=int, default=4)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--block", type=int, default=32)


def _load_model_module(module_file: str):
    module_path = pathlib.Path(module_file).resolve()
    module_spec = importlib.util.spec_from_file_location("cudasim_user_model", module_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"cannot load model module: {module_file}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _build_model(args: argparse.Namespace):
    model_module = _load_model_module(args.module_file)
    model = model_module.build_model(args)
    model.eval()
    return model, model_module.example_input(args)


def _trace(model: torch.nn.Module):
    traced = torch.fx.symbolic_trace(model)
    traced.recompile()
    return traced


def _linear_nodes(traced):
    for node in traced.graph.nodes:
        if node.op != "call_module":
            continue
        module = traced.get_submodule(node.target)
        if isinstance(module, torch.nn.Linear):
            yield node, module


def _relu_nodes(traced):
    for node in traced.graph.nodes:
        if node.op != "call_module":
            continue
        module = traced.get_submodule(node.target)
        if isinstance(module, torch.nn.ReLU):
            yield node, module


def _weighted_nodes(traced):
    for node in traced.graph.nodes:
        if node.op != "call_module":
            continue
        module = traced.get_submodule(node.target)
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            yield node, module


def _batch_norm_nodes(traced):
    for node in traced.graph.nodes:
        if node.op != "call_module":
            continue
        module = traced.get_submodule(node.target)
        if isinstance(module, torch.nn.BatchNorm2d):
            yield node, module


def _max_pool_nodes(traced):
    for node in traced.graph.nodes:
        if node.op != "call_module":
            continue
        module = traced.get_submodule(node.target)
        if isinstance(module, torch.nn.MaxPool2d):
            yield node, module


def _activation_nodes(traced):
    activation_types = (
        torch.nn.GELU,
        torch.nn.SiLU,
        torch.nn.Sigmoid,
        torch.nn.Tanh,
        torch.nn.LeakyReLU,
        torch.nn.Hardswish,
        torch.nn.Hardsigmoid,
        torch.nn.ReLU6,
        torch.nn.Softplus,
        torch.nn.ELU,
        torch.nn.SELU,
        torch.nn.CELU,
        torch.nn.Mish,
        torch.nn.Hardtanh,
        torch.nn.Softsign,
        torch.nn.LogSigmoid,
    )
    for node in traced.graph.nodes:
        if node.op != "call_module":
            continue
        module = traced.get_submodule(node.target)
        if isinstance(module, activation_types):
            yield node, module


def _copy_nodes(traced):
    copy_types = (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Identity, torch.nn.Flatten)
    for node in traced.graph.nodes:
        if node.op != "call_module":
            continue
        module = traced.get_submodule(node.target)
        if isinstance(module, copy_types):
            yield node, module


def _avg_pool_nodes(traced):
    for node in traced.graph.nodes:
        if node.op != "call_module":
            continue
        module = traced.get_submodule(node.target)
        if isinstance(module, torch.nn.AvgPool2d):
            yield node, module


def _adaptive_avg_pool_nodes(traced):
    for node in traced.graph.nodes:
        if node.op != "call_module":
            continue
        module = traced.get_submodule(node.target)
        if isinstance(module, torch.nn.AdaptiveAvgPool2d):
            yield node, module


def _adaptive_max_pool_nodes(traced):
    for node in traced.graph.nodes:
        if node.op != "call_module":
            continue
        module = traced.get_submodule(node.target)
        if isinstance(module, torch.nn.AdaptiveMaxPool2d):
            yield node, module


def _layer_norm_nodes(traced):
    for node in traced.graph.nodes:
        if node.op != "call_module":
            continue
        module = traced.get_submodule(node.target)
        if isinstance(module, torch.nn.LayerNorm):
            yield node, module


def _softmax_nodes(traced):
    for node in traced.graph.nodes:
        if node.op != "call_module":
            continue
        module = traced.get_submodule(node.target)
        if isinstance(module, (torch.nn.Softmax, torch.nn.LogSoftmax)):
            yield node, module


def _activation_mode(module):
    if isinstance(module, torch.nn.GELU):
        if not module.approximate == "none":
            raise RuntimeError("only exact GELU is supported")
        return 0, 0.0
    if isinstance(module, torch.nn.SiLU):
        return 1, 0.0
    if isinstance(module, torch.nn.Sigmoid):
        return 2, 0.0
    if isinstance(module, torch.nn.Tanh):
        return 3, 0.0
    if isinstance(module, torch.nn.LeakyReLU):
        return 4, module.negative_slope
    if isinstance(module, torch.nn.Hardswish):
        return 5, 0.0
    if isinstance(module, torch.nn.Hardsigmoid):
        return 6, 0.0
    if isinstance(module, torch.nn.ReLU6):
        return 7, 0.0
    if isinstance(module, torch.nn.Softplus):
        if module.beta != 1 or module.threshold != 20:
            raise RuntimeError("only default Softplus(beta=1, threshold=20) is supported")
        return 8, 0.0
    if isinstance(module, torch.nn.ELU):
        if module.alpha != 1.0:
            raise RuntimeError("only default ELU(alpha=1) is supported")
        return 9, 0.0
    if isinstance(module, torch.nn.SELU):
        return 10, 0.0
    if isinstance(module, torch.nn.CELU):
        if module.alpha != 1.0:
            raise RuntimeError("only default CELU(alpha=1) is supported")
        return 11, 0.0
    if isinstance(module, torch.nn.Mish):
        return 12, 0.0
    if isinstance(module, torch.nn.Hardtanh):
        if module.min_val != -1.0 or module.max_val != 1.0:
            raise RuntimeError("only default Hardtanh(min_val=-1, max_val=1) is supported")
        return 13, 0.0
    if isinstance(module, torch.nn.Softsign):
        return 14, 0.0
    if isinstance(module, torch.nn.LogSigmoid):
        return 15, 0.0
    raise RuntimeError(f"unsupported activation: {type(module).__name__}")


def _square_pair(module, name):
    value = getattr(module, name)
    if isinstance(value, int):
        value = (value, value)
    if value[0] != value[1]:
        raise RuntimeError(f"unsupported non-square {type(module).__name__} {name}: {value}")
    return value[0]


def _dilation_pair(module):
    value = module.dilation
    if isinstance(value, int):
        value = (value, value)
    return value


def _module_bias(module):
    if module.bias is not None:
        return module.bias
    if isinstance(module, torch.nn.Linear):
        return torch.zeros(module.out_features, dtype=torch.float32)
    return torch.zeros(module.out_channels, dtype=torch.float32)


def _predecessor_name(node):
    value = node.args[0]
    if not isinstance(value, torch.fx.Node):
        raise RuntimeError(f"unsupported input for node {node.name}")
    return value.name


def _binary_input_names(node):
    if len(node.args) < 2:
        raise RuntimeError(f"unsupported binary call: {node.name}")
    left, right = node.args[:2]
    if not isinstance(left, torch.fx.Node) or not isinstance(right, torch.fx.Node):
        raise RuntimeError(f"unsupported binary inputs for node {node.name}")
    return left.name, right.name


def _binary_mode(node):
    if node.target is operator.add:
        return 0
    if node.target is operator.mul:
        return 1
    raise RuntimeError(f"unsupported binary target: {node.target}")


def _cat_input_names(node):
    if not node.args:
        raise RuntimeError(f"unsupported cat call: {node.name}")
    values = node.args[0]
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise RuntimeError("only two-input torch.cat is supported")
    left, right = values
    if not isinstance(left, torch.fx.Node) or not isinstance(right, torch.fx.Node):
        raise RuntimeError(f"unsupported cat inputs for node {node.name}")
    return left.name, right.name


def _cat_dim(node):
    if "dim" in node.kwargs:
        return node.kwargs["dim"]
    if len(node.args) >= 2:
        return node.args[1]
    return 0


def _transpose_dims(node):
    if len(node.args) < 3:
        raise RuntimeError(f"unsupported transpose call: {node.name}")
    dim0, dim1 = node.args[1], node.args[2]
    if not isinstance(dim0, int) or not isinstance(dim1, int):
        raise RuntimeError(f"unsupported transpose dims for {node.name}")
    return dim0, dim1


def _shape_map(traced, example_input):
    shapes = {}
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            shapes[node.name] = list(example_input.shape)
        elif node.op == "output":
            continue
        elif node.op == "call_module":
            module = traced.get_submodule(node.target)
            input_shape = shapes[_predecessor_name(node)]
            if isinstance(module, torch.nn.Linear):
                if len(input_shape) != 2:
                    raise RuntimeError(f"unsupported Linear input shape: {input_shape}")
                shapes[node.name] = [input_shape[0], module.out_features]
            elif isinstance(module, torch.nn.Conv2d):
                if len(input_shape) != 4:
                    raise RuntimeError(f"unsupported Conv2d input shape: {input_shape}")
                if module.groups != 1:
                    raise RuntimeError("unsupported grouped Conv2d")
                if module.padding_mode != "zeros":
                    raise RuntimeError("unsupported non-zero Conv2d padding mode")
                if _dilation_pair(module) != (1, 1):
                    raise RuntimeError("unsupported dilated Conv2d")
                kernel_h = _square_pair(module, "kernel_size")
                kernel_w = kernel_h
                stride_h = _square_pair(module, "stride")
                stride_w = stride_h
                pad_h = _square_pair(module, "padding")
                pad_w = pad_h
                dilation_h, dilation_w = 1, 1
                out_h = (
                    input_shape[2] + 2 * pad_h - dilation_h * (kernel_h - 1) - 1
                ) // stride_h + 1
                out_w = (
                    input_shape[3] + 2 * pad_w - dilation_w * (kernel_w - 1) - 1
                ) // stride_w + 1
                shapes[node.name] = [
                    input_shape[0],
                    module.out_channels,
                    out_h,
                    out_w,
                ]
            elif isinstance(module, torch.nn.BatchNorm2d):
                if len(input_shape) != 4:
                    raise RuntimeError(f"unsupported BatchNorm2d input shape: {input_shape}")
                if not module.track_running_stats:
                    raise RuntimeError("BatchNorm2d requires track_running_stats=True")
                shapes[node.name] = input_shape
            elif isinstance(module, torch.nn.MaxPool2d):
                if len(input_shape) != 4:
                    raise RuntimeError(f"unsupported MaxPool2d input shape: {input_shape}")
                if module.ceil_mode:
                    raise RuntimeError("unsupported ceil_mode MaxPool2d")
                if module.return_indices:
                    raise RuntimeError("unsupported return_indices MaxPool2d")
                if _dilation_pair(module) != (1, 1):
                    raise RuntimeError("unsupported dilated MaxPool2d")
                kernel_h = _square_pair(module, "kernel_size")
                kernel_w = kernel_h
                stride_h = _square_pair(module, "stride")
                stride_w = stride_h
                pad_h = _square_pair(module, "padding")
                pad_w = pad_h
                out_h = (input_shape[2] + 2 * pad_h - (kernel_h - 1) - 1) // stride_h + 1
                out_w = (input_shape[3] + 2 * pad_w - (kernel_w - 1) - 1) // stride_w + 1
                shapes[node.name] = [input_shape[0], input_shape[1], out_h, out_w]
            elif isinstance(module, torch.nn.AvgPool2d):
                if len(input_shape) != 4:
                    raise RuntimeError(f"unsupported AvgPool2d input shape: {input_shape}")
                if module.ceil_mode:
                    raise RuntimeError("unsupported ceil_mode AvgPool2d")
                if not module.count_include_pad:
                    raise RuntimeError("unsupported AvgPool2d count_include_pad=False")
                if module.divisor_override is not None:
                    raise RuntimeError("unsupported AvgPool2d divisor_override")
                kernel_h = _square_pair(module, "kernel_size")
                kernel_w = kernel_h
                stride_h = _square_pair(module, "stride")
                stride_w = stride_h
                pad_h = _square_pair(module, "padding")
                pad_w = pad_h
                out_h = (input_shape[2] + 2 * pad_h - (kernel_h - 1) - 1) // stride_h + 1
                out_w = (input_shape[3] + 2 * pad_w - (kernel_w - 1) - 1) // stride_w + 1
                shapes[node.name] = [input_shape[0], input_shape[1], out_h, out_w]
            elif isinstance(module, torch.nn.AdaptiveAvgPool2d):
                if len(input_shape) != 4:
                    raise RuntimeError(f"unsupported AdaptiveAvgPool2d input shape: {input_shape}")
                if tuple(module.output_size) != (1, 1):
                    raise RuntimeError("only AdaptiveAvgPool2d((1, 1)) is supported")
                shapes[node.name] = [input_shape[0], input_shape[1], 1, 1]
            elif isinstance(module, torch.nn.AdaptiveMaxPool2d):
                if len(input_shape) != 4:
                    raise RuntimeError(f"unsupported AdaptiveMaxPool2d input shape: {input_shape}")
                if tuple(module.output_size) != (1, 1):
                    raise RuntimeError("only AdaptiveMaxPool2d((1, 1)) is supported")
                if module.return_indices:
                    raise RuntimeError("unsupported return_indices AdaptiveMaxPool2d")
                shapes[node.name] = [input_shape[0], input_shape[1], 1, 1]
            elif isinstance(module, torch.nn.Flatten):
                if module.start_dim != 1 or module.end_dim != -1:
                    raise RuntimeError("only Flatten(start_dim=1, end_dim=-1) is supported")
                flat = 1
                for dim in input_shape[1:]:
                    flat *= dim
                shapes[node.name] = [input_shape[0], flat]
            elif isinstance(
                module,
                (
                    torch.nn.GELU,
                    torch.nn.SiLU,
                    torch.nn.Sigmoid,
                    torch.nn.Tanh,
                    torch.nn.LeakyReLU,
                    torch.nn.Hardswish,
                    torch.nn.Hardsigmoid,
                    torch.nn.ReLU6,
                    torch.nn.Softplus,
                    torch.nn.ELU,
                    torch.nn.SELU,
                    torch.nn.CELU,
                    torch.nn.Mish,
                    torch.nn.Hardtanh,
                    torch.nn.Softsign,
                    torch.nn.LogSigmoid,
                    torch.nn.Dropout,
                    torch.nn.Dropout2d,
                    torch.nn.Identity,
                ),
            ):
                shapes[node.name] = input_shape
            elif isinstance(module, torch.nn.LayerNorm):
                if len(input_shape) < 2:
                    raise RuntimeError(f"unsupported LayerNorm input shape: {input_shape}")
                if list(module.normalized_shape) != [input_shape[-1]]:
                    raise RuntimeError("only last-dim LayerNorm is supported")
                shapes[node.name] = input_shape
            elif isinstance(module, (torch.nn.Softmax, torch.nn.LogSoftmax)):
                if module.dim not in (-1, len(input_shape) - 1):
                    raise RuntimeError("only last-dim softmax is supported")
                shapes[node.name] = input_shape
            elif isinstance(module, torch.nn.ReLU):
                shapes[node.name] = input_shape
            else:
                raise RuntimeError(f"unsupported module in traced graph: {type(module).__name__}")
        elif node.op == "call_function" and node.target in (operator.add, operator.mul):
            left_name, right_name = _binary_input_names(node)
            if shapes[left_name] != shapes[right_name]:
                raise RuntimeError(f"mismatched binary input shapes for {node.name}")
            shapes[node.name] = shapes[left_name]
        elif node.op == "call_function" and node.target is torch.cat:
            left_name, right_name = _cat_input_names(node)
            dim = _cat_dim(node)
            left_shape = shapes[left_name]
            right_shape = shapes[right_name]
            if dim not in (-1, len(left_shape) - 1):
                raise RuntimeError("only last-dim torch.cat is supported")
            if left_shape[:-1] != right_shape[:-1]:
                raise RuntimeError(f"mismatched torch.cat shapes for {node.name}")
            shapes[node.name] = left_shape[:-1] + [left_shape[-1] + right_shape[-1]]
        elif node.op == "call_function" and node.target is torch.matmul:
            left_name, right_name = _binary_input_names(node)
            left_shape = shapes[left_name]
            right_shape = shapes[right_name]
            if len(left_shape) != 2 or len(right_shape) != 2:
                raise RuntimeError("only 2D torch.matmul is supported")
            if left_shape[1] != right_shape[0]:
                raise RuntimeError(f"mismatched torch.matmul shapes for {node.name}")
            shapes[node.name] = [left_shape[0], right_shape[1]]
        elif node.op == "call_function" and node.target is torch.flatten:
            input_name = _predecessor_name(node)
            input_shape = shapes[input_name]
            start_dim = node.args[1] if len(node.args) >= 2 else 0
            if start_dim != 1:
                raise RuntimeError("only torch.flatten(x, start_dim=1) is supported")
            flat = 1
            for dim in input_shape[1:]:
                flat *= dim
            shapes[node.name] = [input_shape[0], flat]
        elif node.op == "call_method" and node.target == "transpose":
            input_name = _predecessor_name(node)
            input_shape = shapes[input_name]
            if len(input_shape) != 2:
                raise RuntimeError("only 2D transpose is supported")
            dim0, dim1 = _transpose_dims(node)
            if dim0 < 0:
                dim0 += 2
            if dim1 < 0:
                dim1 += 2
            if (dim0, dim1) not in ((0, 1), (1, 0)):
                raise RuntimeError("only 2D row/column transpose is supported")
            shapes[node.name] = [input_shape[1], input_shape[0]]
        else:
            raise RuntimeError(f"unsupported traced node: {node.op} {node.name}")
    return shapes


def _graph_program(traced, shapes):
    program = []
    for node in traced.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        if node.op == "call_function" and node.target in (operator.add, operator.mul):
            left_name, right_name = _binary_input_names(node)
            program.append(
                {
                    "kind": "binary",
                    "name": node.name,
                    "inputs": (left_name, right_name),
                    "mode": _binary_mode(node),
                    "shape": shapes[node.name],
                }
            )
            continue
        if node.op == "call_function" and node.target is torch.cat:
            left_name, right_name = _cat_input_names(node)
            program.append(
                {
                    "kind": "cat",
                    "name": node.name,
                    "inputs": (left_name, right_name),
                    "shape": shapes[node.name],
                }
            )
            continue
        if node.op == "call_function" and node.target is torch.matmul:
            left_name, right_name = _binary_input_names(node)
            program.append(
                {
                    "kind": "matmul",
                    "name": node.name,
                    "inputs": (left_name, right_name),
                    "shape": shapes[node.name],
                }
            )
            continue
        if node.op == "call_function" and node.target is torch.flatten:
            input_name = _predecessor_name(node)
            program.append(
                {
                    "kind": "copy",
                    "name": node.name,
                    "input": input_name,
                    "shape": shapes[node.name],
                }
            )
            continue
        if node.op == "call_method" and node.target == "transpose":
            input_name = _predecessor_name(node)
            program.append(
                {
                    "kind": "transpose",
                    "name": node.name,
                    "input": input_name,
                    "shape": shapes[node.name],
                }
            )
            continue
        module = traced.get_submodule(node.target)
        input_name = _predecessor_name(node)
        if isinstance(module, torch.nn.Linear):
            program.append(
                {
                    "kind": "linear",
                    "name": node.name,
                    "input": input_name,
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "shape": shapes[node.name],
                }
            )
        elif isinstance(module, torch.nn.Conv2d):
            kernel_h = _square_pair(module, "kernel_size")
            kernel_w = kernel_h
            stride_h = _square_pair(module, "stride")
            stride_w = stride_h
            pad_h = _square_pair(module, "padding")
            pad_w = pad_h
            program.append(
                {
                    "kind": "conv2d",
                    "name": node.name,
                    "input": input_name,
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "kernel_h": kernel_h,
                    "kernel_w": kernel_w,
                    "stride_h": stride_h,
                    "stride_w": stride_w,
                    "pad_h": pad_h,
                    "pad_w": pad_w,
                    "input_shape": shapes[_predecessor_name(node)],
                    "shape": shapes[node.name],
                }
            )
        elif isinstance(module, torch.nn.BatchNorm2d):
            program.append(
                {
                    "kind": "batchnorm2d",
                    "name": node.name,
                    "input": input_name,
                    "channels": module.num_features,
                    "eps": module.eps,
                    "input_shape": shapes[_predecessor_name(node)],
                    "shape": shapes[node.name],
                }
            )
        elif isinstance(module, torch.nn.MaxPool2d):
            kernel_h = _square_pair(module, "kernel_size")
            kernel_w = kernel_h
            stride_h = _square_pair(module, "stride")
            stride_w = stride_h
            pad_h = _square_pair(module, "padding")
            pad_w = pad_h
            program.append(
                {
                    "kind": "maxpool2d",
                    "name": node.name,
                    "input": input_name,
                    "kernel_h": kernel_h,
                    "kernel_w": kernel_w,
                    "stride_h": stride_h,
                    "stride_w": stride_w,
                    "pad_h": pad_h,
                    "pad_w": pad_w,
                    "input_shape": shapes[_predecessor_name(node)],
                    "shape": shapes[node.name],
                }
            )
        elif isinstance(module, torch.nn.AvgPool2d):
            kernel_h = _square_pair(module, "kernel_size")
            kernel_w = kernel_h
            stride_h = _square_pair(module, "stride")
            stride_w = stride_h
            pad_h = _square_pair(module, "padding")
            pad_w = pad_h
            program.append(
                {
                    "kind": "avgpool2d",
                    "name": node.name,
                    "input": input_name,
                    "kernel_h": kernel_h,
                    "kernel_w": kernel_w,
                    "stride_h": stride_h,
                    "stride_w": stride_w,
                    "pad_h": pad_h,
                    "pad_w": pad_w,
                    "input_shape": shapes[_predecessor_name(node)],
                    "shape": shapes[node.name],
                }
            )
        elif isinstance(module, torch.nn.AdaptiveAvgPool2d):
            program.append(
                {
                    "kind": "adaptive_avgpool2d",
                    "name": node.name,
                    "input": input_name,
                    "input_shape": shapes[_predecessor_name(node)],
                    "shape": shapes[node.name],
                }
            )
        elif isinstance(module, torch.nn.AdaptiveMaxPool2d):
            program.append(
                {
                    "kind": "adaptive_maxpool2d",
                    "name": node.name,
                    "input": input_name,
                    "input_shape": shapes[_predecessor_name(node)],
                    "shape": shapes[node.name],
                }
            )
        elif isinstance(
            module,
            (
                torch.nn.GELU,
                torch.nn.SiLU,
                torch.nn.Sigmoid,
                torch.nn.Tanh,
                torch.nn.LeakyReLU,
                torch.nn.Hardswish,
                torch.nn.Hardsigmoid,
                torch.nn.ReLU6,
                torch.nn.Softplus,
                torch.nn.ELU,
                torch.nn.SELU,
                torch.nn.CELU,
                torch.nn.Mish,
                torch.nn.Hardtanh,
                torch.nn.Softsign,
                torch.nn.LogSigmoid,
            ),
        ):
            mode, negative_slope = _activation_mode(module)
            program.append(
                {
                    "kind": "activation",
                    "name": node.name,
                    "input": input_name,
                    "mode": mode,
                    "negative_slope": negative_slope,
                    "shape": shapes[node.name],
                }
            )
        elif isinstance(
            module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Identity, torch.nn.Flatten)
        ):
            program.append(
                {
                    "kind": "copy",
                    "name": node.name,
                    "input": input_name,
                    "shape": shapes[node.name],
                }
            )
        elif isinstance(module, torch.nn.LayerNorm):
            program.append(
                {
                    "kind": "layernorm",
                    "name": node.name,
                    "input": input_name,
                    "normalized_size": shapes[input_name][-1],
                    "eps": module.eps,
                    "shape": shapes[node.name],
                }
            )
        elif isinstance(module, (torch.nn.Softmax, torch.nn.LogSoftmax)):
            program.append(
                {
                    "kind": "softmax",
                    "name": node.name,
                    "input": input_name,
                    "normalized_size": shapes[input_name][-1],
                    "log": isinstance(module, torch.nn.LogSoftmax),
                    "shape": shapes[node.name],
                }
            )
        elif isinstance(module, torch.nn.ReLU):
            program.append(
                {
                    "kind": "relu",
                    "name": node.name,
                    "input": input_name,
                    "shape": shapes[node.name],
                }
            )
        else:
            raise RuntimeError(f"unsupported module in traced graph: {type(module).__name__}")
    return program


def run_cpu(args: argparse.Namespace, output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1234)
    model, example_input = _build_model(args)
    with torch.no_grad():
        output = model(example_input)

    torch.save(
        {
            "x": example_input,
            "output": output,
            "shape": tuple(output.shape),
            "dtype": str(output.dtype),
        },
        output_dir / "reference.pt",
    )
    write_floats(output_dir / "input.bin", example_input)
    write_floats(output_dir / "cpu_output.bin", output)

    traced = _trace(model)
    for index, (_, module) in enumerate(_weighted_nodes(traced)):
        write_floats(output_dir / f"weight{index}.bin", module.weight)
        write_floats(output_dir / f"bias{index}.bin", _module_bias(module))
    for index, (_, module) in enumerate(_batch_norm_nodes(traced)):
        weight = module.weight if module.weight is not None else torch.ones(module.num_features)
        bias = module.bias if module.bias is not None else torch.zeros(module.num_features)
        mean = (
            module.running_mean
            if module.running_mean is not None
            else torch.zeros(module.num_features)
        )
        var = (
            module.running_var
            if module.running_var is not None
            else torch.ones(module.num_features)
        )
        write_floats(output_dir / f"bn_weight{index}.bin", weight)
        write_floats(output_dir / f"bn_bias{index}.bin", bias)
        write_floats(output_dir / f"bn_mean{index}.bin", mean)
        write_floats(output_dir / f"bn_var{index}.bin", var)
    for index, (_, module) in enumerate(_layer_norm_nodes(traced)):
        normalized_size = module.normalized_shape[0]
        weight = module.weight if module.weight is not None else torch.ones(normalized_size)
        bias = module.bias if module.bias is not None else torch.zeros(normalized_size)
        write_floats(output_dir / f"ln_weight{index}.bin", weight)
        write_floats(output_dir / f"ln_bias{index}.bin", bias)


def native_arguments(args: argparse.Namespace, output_dir: pathlib.Path) -> list[str]:
    model, _ = _build_model(args)
    traced = _trace(model)
    weighted = list(_weighted_nodes(traced))
    batch_norms = list(_batch_norm_nodes(traced))
    layer_norms = list(_layer_norm_nodes(traced))
    arguments = [
        "--input",
        str(output_dir / "input.bin"),
        "--output",
        str(output_dir / "gpu_output.bin"),
    ]
    for index, _ in enumerate(weighted):
        arguments.extend(
            [
                f"--weight{index}",
                str(output_dir / f"weight{index}.bin"),
                f"--bias{index}",
                str(output_dir / f"bias{index}.bin"),
            ]
        )
    for index, _ in enumerate(batch_norms):
        arguments.extend(
            [
                f"--bn-weight{index}",
                str(output_dir / f"bn_weight{index}.bin"),
                f"--bn-bias{index}",
                str(output_dir / f"bn_bias{index}.bin"),
                f"--bn-mean{index}",
                str(output_dir / f"bn_mean{index}.bin"),
                f"--bn-var{index}",
                str(output_dir / f"bn_var{index}.bin"),
            ]
        )
    for index, _ in enumerate(layer_norms):
        arguments.extend(
            [
                f"--ln-weight{index}",
                str(output_dir / f"ln_weight{index}.bin"),
                f"--ln-bias{index}",
                str(output_dir / f"ln_bias{index}.bin"),
            ]
        )
    return arguments


def _buffer_name(node_name):
    return "d_" + node_name.replace(".", "_")


def _numel(shape):
    count = 1
    for dim in shape:
        count *= dim
    return count


def _linear_index(linears, op):
    return next(index for index, candidate in enumerate(linears) if candidate is op)


def _conv_index(convs, op):
    return next(index for index, candidate in enumerate(convs) if candidate is op)


def _relu_index(relus, op):
    return next(index for index, candidate in enumerate(relus) if candidate is op)


def _activation_index(activations, op):
    return next(index for index, candidate in enumerate(activations) if candidate is op)


def _copy_index(copies, op):
    return next(index for index, candidate in enumerate(copies) if candidate is op)


def _batch_norm_index(batch_norms, op):
    return next(index for index, candidate in enumerate(batch_norms) if candidate is op)


def _max_pool_index(max_pools, op):
    return next(index for index, candidate in enumerate(max_pools) if candidate is op)


def _avg_pool_index(avg_pools, op):
    return next(index for index, candidate in enumerate(avg_pools) if candidate is op)


def _adaptive_avg_pool_index(adaptive_avg_pools, op):
    return next(index for index, candidate in enumerate(adaptive_avg_pools) if candidate is op)


def _adaptive_max_pool_index(adaptive_max_pools, op):
    return next(index for index, candidate in enumerate(adaptive_max_pools) if candidate is op)


def _binary_index(binaries, op):
    return next(index for index, candidate in enumerate(binaries) if candidate is op)


def _cat_index(cats, op):
    return next(index for index, candidate in enumerate(cats) if candidate is op)


def _transpose_index(transposes, op):
    return next(index for index, candidate in enumerate(transposes) if candidate is op)


def _matmul_index(matmuls, op):
    return next(index for index, candidate in enumerate(matmuls) if candidate is op)


def _layer_norm_index(layer_norms, op):
    return next(index for index, candidate in enumerate(layer_norms) if candidate is op)


def _softmax_index(softmaxes, op):
    return next(index for index, candidate in enumerate(softmaxes) if candidate is op)


def _weighted_index(weighted_ops, op):
    return next(index for index, candidate in enumerate(weighted_ops) if candidate is op)


def generate_native_runner(args: argparse.Namespace, output_path: pathlib.Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-capable PyTorch is required to generate Triton PTX")

    model, example_input = _build_model(args)
    traced = _trace(model)
    shapes = _shape_map(traced, example_input)
    program = _graph_program(traced, shapes)
    linears = [op for op in program if op["kind"] == "linear"]
    convs = [op for op in program if op["kind"] == "conv2d"]
    relus = [op for op in program if op["kind"] == "relu"]
    activations = [op for op in program if op["kind"] == "activation"]
    copies = [op for op in program if op["kind"] == "copy"]
    batch_norms = [op for op in program if op["kind"] == "batchnorm2d"]
    max_pools = [op for op in program if op["kind"] == "maxpool2d"]
    avg_pools = [op for op in program if op["kind"] == "avgpool2d"]
    adaptive_avg_pools = [op for op in program if op["kind"] == "adaptive_avgpool2d"]
    adaptive_max_pools = [op for op in program if op["kind"] == "adaptive_maxpool2d"]
    binaries = [op for op in program if op["kind"] == "binary"]
    cats = [op for op in program if op["kind"] == "cat"]
    transposes = [op for op in program if op["kind"] == "transpose"]
    matmuls = [op for op in program if op["kind"] == "matmul"]
    layer_norms = [op for op in program if op["kind"] == "layernorm"]
    softmaxes = [op for op in program if op["kind"] == "softmax"]
    weighted = [op for op in program if op["kind"] in ("linear", "conv2d")]

    ptx_by_key = {}
    for index, op in enumerate(linears):
        m = op["shape"][0]
        k = op["in_features"]
        n = op["out_features"]
        x = torch.randn(m, k, device="cuda", dtype=torch.float32)
        w = torch.randn(n, k, device="cuda", dtype=torch.float32)
        b = torch.randn(n, device="cuda", dtype=torch.float32)
        out = torch.empty(m, n, device="cuda", dtype=torch.float32)
        ptx = compile_kernel(
            linear_kernel,
            (m, n),
            (x, w, b, out, m, n),
            {"k": k, "num_warps": 1},
        ).replace("linear_kernel", f"linear_{index}_kernel")
        ptx_by_key[f"linear_{index}"] = ptx

    for index, op in enumerate(convs):
        n, c, h, w = op["input_shape"]
        k = op["out_channels"]
        r = op["kernel_h"]
        s = op["kernel_w"]
        stride = op["stride_h"]
        pad = op["pad_h"]
        out_h = op["shape"][2]
        out_w = op["shape"][3]
        x = torch.randn(n, c, h, w, device="cuda", dtype=torch.float32)
        weight = torch.randn(k, c, r, s, device="cuda", dtype=torch.float32)
        bias = torch.randn(k, device="cuda", dtype=torch.float32)
        out = torch.empty(n, k, out_h, out_w, device="cuda", dtype=torch.float32)
        ptx = compile_kernel(
            conv2d_kernel,
            (n * k, out_h, out_w),
            (x, weight, bias, out),
            {
                "C": c,
                "H": h,
                "W": w,
                "K": k,
                "R": r,
                "S": s,
                "stride": stride,
                "pad": pad,
                "num_warps": 1,
            },
        ).replace("conv2d_kernel", f"conv2d_{index}_kernel")
        ptx_by_key[f"conv2d_{index}"] = ptx

    for index, op in enumerate(relus):
        count = _numel(op["shape"])
        input_tensor = torch.empty(count, device="cuda", dtype=torch.float32)
        output_tensor = torch.empty(count, device="cuda", dtype=torch.float32)
        ptx = compile_kernel(
            relu_kernel,
            (1,),
            (input_tensor, output_tensor, count),
            {"BLOCK": args.block, "num_warps": 1},
        ).replace("relu_kernel", f"relu_{index}_kernel")
        ptx_by_key[f"relu_{index}"] = ptx

    for index, op in enumerate(activations):
        count = _numel(op["shape"])
        input_tensor = torch.empty(count, device="cuda", dtype=torch.float32)
        output_tensor = torch.empty(count, device="cuda", dtype=torch.float32)
        ptx = compile_kernel(
            activation_kernel,
            (1,),
            (input_tensor, output_tensor, count),
            {
                "BLOCK": args.block,
                "mode": op["mode"],
                "negative_slope": op["negative_slope"],
                "num_warps": 1,
            },
        ).replace("activation_kernel", f"activation_{index}_kernel")
        ptx_by_key[f"activation_{index}"] = ptx

    for index, op in enumerate(copies):
        count = _numel(op["shape"])
        input_tensor = torch.empty(count, device="cuda", dtype=torch.float32)
        output_tensor = torch.empty(count, device="cuda", dtype=torch.float32)
        ptx = compile_kernel(
            copy_kernel,
            (1,),
            (input_tensor, output_tensor, count),
            {"BLOCK": args.block, "num_warps": 1},
        ).replace("copy_kernel", f"copy_{index}_kernel")
        ptx_by_key[f"copy_{index}"] = ptx

    for index, op in enumerate(batch_norms):
        n, c, h, w = op["input_shape"]
        count = n * c * h * w
        x = torch.randn(count, device="cuda", dtype=torch.float32)
        weight = torch.randn(c, device="cuda", dtype=torch.float32)
        bias = torch.randn(c, device="cuda", dtype=torch.float32)
        mean = torch.randn(c, device="cuda", dtype=torch.float32)
        var = torch.rand(c, device="cuda", dtype=torch.float32) + 1.0
        out = torch.empty(count, device="cuda", dtype=torch.float32)
        ptx = compile_kernel(
            batchnorm2d_kernel,
            (count,),
            (x, weight, bias, mean, var, out),
            {
                "C": c,
                "H": h,
                "W": w,
                "eps": op["eps"],
                "num_warps": 1,
            },
        ).replace("batchnorm2d_kernel", f"batchnorm2d_{index}_kernel")
        ptx_by_key[f"batchnorm2d_{index}"] = ptx

    for index, op in enumerate(max_pools):
        n, c, h, w = op["input_shape"]
        out_h, out_w = op["shape"][2], op["shape"][3]
        x = torch.randn(n, c, h, w, device="cuda", dtype=torch.float32)
        out = torch.empty(n, c, out_h, out_w, device="cuda", dtype=torch.float32)
        ptx = compile_kernel(
            maxpool2d_kernel,
            (n * c, out_h, out_w),
            (x, out),
            {
                "C": c,
                "H": h,
                "W": w,
                "KH": op["kernel_h"],
                "KW": op["kernel_w"],
                "stride": op["stride_h"],
                "pad": op["pad_h"],
                "num_warps": 1,
            },
        ).replace("maxpool2d_kernel", f"maxpool2d_{index}_kernel")
        ptx_by_key[f"maxpool2d_{index}"] = ptx

    for index, op in enumerate(avg_pools):
        n, c, h, w = op["input_shape"]
        out_h, out_w = op["shape"][2], op["shape"][3]
        x = torch.randn(n, c, h, w, device="cuda", dtype=torch.float32)
        out = torch.empty(n, c, out_h, out_w, device="cuda", dtype=torch.float32)
        ptx = compile_kernel(
            avgpool2d_kernel,
            (n * c, out_h, out_w),
            (x, out),
            {
                "C": c,
                "H": h,
                "W": w,
                "KH": op["kernel_h"],
                "KW": op["kernel_w"],
                "stride": op["stride_h"],
                "pad": op["pad_h"],
                "num_warps": 1,
            },
        ).replace("avgpool2d_kernel", f"avgpool2d_{index}_kernel")
        ptx_by_key[f"avgpool2d_{index}"] = ptx

    for index, op in enumerate(adaptive_avg_pools):
        n, c, h, w = op["input_shape"]
        x = torch.randn(n, c, h, w, device="cuda", dtype=torch.float32)
        out = torch.empty(n * c, device="cuda", dtype=torch.float32)
        ptx = compile_kernel(
            adaptive_avgpool2d_1x1_kernel,
            (n * c,),
            (x, out),
            {"H": h, "W": w, "num_warps": 1},
        ).replace(
            "adaptive_avgpool2d_1x1_kernel",
            f"adaptive_avgpool2d_{index}_kernel",
        )
        ptx_by_key[f"adaptive_avgpool2d_{index}"] = ptx

    for index, op in enumerate(adaptive_max_pools):
        n, c, h, w = op["input_shape"]
        x = torch.randn(n, c, h, w, device="cuda", dtype=torch.float32)
        out = torch.empty(n * c, device="cuda", dtype=torch.float32)
        ptx = compile_kernel(
            adaptive_maxpool2d_1x1_kernel,
            (n * c,),
            (x, out),
            {"H": h, "W": w, "num_warps": 1},
        ).replace(
            "adaptive_maxpool2d_1x1_kernel",
            f"adaptive_maxpool2d_{index}_kernel",
        )
        ptx_by_key[f"adaptive_maxpool2d_{index}"] = ptx

    for index, op in enumerate(binaries):
        count = _numel(op["shape"])
        a = torch.empty(count, device="cuda", dtype=torch.float32)
        b = torch.empty(count, device="cuda", dtype=torch.float32)
        out = torch.empty(count, device="cuda", dtype=torch.float32)
        ptx = compile_kernel(
            binary_kernel,
            (1,),
            (a, b, out, count),
            {
                "BLOCK": args.block,
                "mode": op["mode"],
                "num_warps": 1,
            },
        ).replace("binary_kernel", f"binary_{index}_kernel")
        ptx_by_key[f"binary_{index}"] = ptx

    for index, op in enumerate(cats):
        left_shape = shapes[op["inputs"][0]]
        right_shape = shapes[op["inputs"][1]]
        a_inner = left_shape[-1]
        b_inner = right_shape[-1]
        count = _numel(op["shape"])
        a = torch.empty(count, device="cuda", dtype=torch.float32)
        b = torch.empty(count, device="cuda", dtype=torch.float32)
        out = torch.empty(count, device="cuda", dtype=torch.float32)
        ptx = compile_kernel(
            concat_last_kernel,
            (1,),
            (a, b, out, count),
            {
                "BLOCK": args.block,
                "A_INNER": a_inner,
                "B_INNER": b_inner,
                "num_warps": 1,
            },
        ).replace("concat_last_kernel", f"cat_{index}_kernel")
        ptx_by_key[f"cat_{index}"] = ptx

    for index, op in enumerate(transposes):
        input_shape = shapes[op["input"]]
        m, n = input_shape
        count = m * n
        x = torch.empty(count, device="cuda", dtype=torch.float32)
        out = torch.empty(count, device="cuda", dtype=torch.float32)
        ptx = compile_kernel(
            transpose_kernel,
            (1,),
            (x, out),
            {"M": m, "N": n, "BLOCK": args.block, "num_warps": 1},
        ).replace("transpose_kernel", f"transpose_{index}_kernel")
        ptx_by_key[f"transpose_{index}"] = ptx

    for index, op in enumerate(matmuls):
        left_shape = shapes[op["inputs"][0]]
        right_shape = shapes[op["inputs"][1]]
        m = left_shape[0]
        k = left_shape[1]
        n = right_shape[1]
        a = torch.randn(m, k, device="cuda", dtype=torch.float32)
        b = torch.randn(k, n, device="cuda", dtype=torch.float32)
        out = torch.empty(m, n, device="cuda", dtype=torch.float32)
        ptx = compile_kernel(
            matmul_kernel,
            (m, n),
            (a, b, out),
            {"K": k, "N": n, "num_warps": 1},
        ).replace("matmul_kernel", f"matmul_{index}_kernel")
        ptx_by_key[f"matmul_{index}"] = ptx

    for index, op in enumerate(layer_norms):
        n = op["normalized_size"]
        count = _numel(op["shape"])
        x = torch.randn(count, device="cuda", dtype=torch.float32)
        weight = torch.randn(n, device="cuda", dtype=torch.float32)
        bias = torch.randn(n, device="cuda", dtype=torch.float32)
        out = torch.empty(count, device="cuda", dtype=torch.float32)
        ptx = compile_kernel(
            layernorm_kernel,
            (count,),
            (x, weight, bias, out),
            {"N": n, "eps": op["eps"], "num_warps": 1},
        ).replace("layernorm_kernel", f"layernorm_{index}_kernel")
        ptx_by_key[f"layernorm_{index}"] = ptx

    for index, op in enumerate(softmaxes):
        n = op["normalized_size"]
        count = _numel(op["shape"])
        x = torch.randn(count, device="cuda", dtype=torch.float32)
        out = torch.empty(count, device="cuda", dtype=torch.float32)
        ptx = compile_kernel(
            softmax_kernel,
            (count,),
            (x, out),
            {"N": n, "log": op["log"], "num_warps": 1},
        ).replace("softmax_kernel", f"softmax_{index}_kernel")
        ptx_by_key[f"softmax_{index}"] = ptx

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _generate_cpp(
            args,
            traced,
            program,
            linears,
            convs,
            relus,
            activations,
            copies,
            batch_norms,
            max_pools,
            avg_pools,
            adaptive_avg_pools,
            adaptive_max_pools,
            binaries,
            cats,
            transposes,
            matmuls,
            layer_norms,
            softmaxes,
            weighted,
            shapes,
            ptx_by_key,
        )
    )


def _generate_cpp(
    args,
    traced,
    program,
    linears,
    convs,
    relus,
    activations,
    copies,
    batch_norms,
    max_pools,
    avg_pools,
    adaptive_avg_pools,
    adaptive_max_pools,
    binaries,
    cats,
    transposes,
    matmuls,
    layer_norms,
    softmaxes,
    weighted,
    shapes,
    ptx_by_key,
):
    placeholder = next(node for node in traced.graph.nodes if node.op == "placeholder")
    output_node = next(node for node in traced.graph.nodes if node.op == "output")
    final_node = output_node.args[0]
    if not isinstance(final_node, torch.fx.Node):
        raise RuntimeError("traced model must return a tensor")

    function_loads = []
    for index, _ in enumerate(linears):
        function_loads.append(f"""  CUfunction linear{index}_fn = load_function(
      R"CUPTX({ptx_by_key[f"linear_{index}"]})CUPTX", "linear_{index}_kernel");""")
    for index, _ in enumerate(convs):
        function_loads.append(f"""  CUfunction conv{index}_fn = load_function(
      R"CUPTX({ptx_by_key[f"conv2d_{index}"]})CUPTX", "conv2d_{index}_kernel");""")
    for index, _ in enumerate(relus):
        function_loads.append(f"""  CUfunction relu{index}_fn = load_function(
      R"CUPTX({ptx_by_key[f"relu_{index}"]})CUPTX", "relu_{index}_kernel");""")
    for index, _ in enumerate(activations):
        function_loads.append(f"""  CUfunction activation{index}_fn = load_function(
      R"CUPTX({ptx_by_key[f"activation_{index}"]})CUPTX", "activation_{index}_kernel");""")
    for index, _ in enumerate(copies):
        function_loads.append(f"""  CUfunction copy{index}_fn = load_function(
      R"CUPTX({ptx_by_key[f"copy_{index}"]})CUPTX", "copy_{index}_kernel");""")
    for index, _ in enumerate(batch_norms):
        function_loads.append(f"""  CUfunction batchnorm{index}_fn = load_function(
      R"CUPTX({ptx_by_key[f"batchnorm2d_{index}"]})CUPTX", "batchnorm2d_{index}_kernel");""")
    for index, _ in enumerate(max_pools):
        function_loads.append(f"""  CUfunction maxpool{index}_fn = load_function(
      R"CUPTX({ptx_by_key[f"maxpool2d_{index}"]})CUPTX", "maxpool2d_{index}_kernel");""")
    for index, _ in enumerate(avg_pools):
        function_loads.append(f"""  CUfunction avgpool{index}_fn = load_function(
      R"CUPTX({ptx_by_key[f"avgpool2d_{index}"]})CUPTX", "avgpool2d_{index}_kernel");""")
    for index, _ in enumerate(adaptive_avg_pools):
        function_loads.append(
            f"""  CUfunction adaptive_avgpool{index}_fn = load_function(
      R"CUPTX({ptx_by_key[f"adaptive_avgpool2d_{index}"]})CUPTX", "adaptive_avgpool2d_{index}_kernel");"""
        )
    for index, _ in enumerate(adaptive_max_pools):
        function_loads.append(
            f"""  CUfunction adaptive_maxpool{index}_fn = load_function(
      R"CUPTX({ptx_by_key[f"adaptive_maxpool2d_{index}"]})CUPTX", "adaptive_maxpool2d_{index}_kernel");"""
        )
    for index, _ in enumerate(binaries):
        function_loads.append(f"""  CUfunction binary{index}_fn = load_function(
      R"CUPTX({ptx_by_key[f"binary_{index}"]})CUPTX", "binary_{index}_kernel");""")
    for index, _ in enumerate(cats):
        function_loads.append(f"""  CUfunction cat{index}_fn = load_function(
      R"CUPTX({ptx_by_key[f"cat_{index}"]})CUPTX", "cat_{index}_kernel");""")
    for index, _ in enumerate(transposes):
        function_loads.append(f"""  CUfunction transpose{index}_fn = load_function(
      R"CUPTX({ptx_by_key[f"transpose_{index}"]})CUPTX", "transpose_{index}_kernel");""")
    for index, _ in enumerate(matmuls):
        function_loads.append(f"""  CUfunction matmul{index}_fn = load_function(
      R"CUPTX({ptx_by_key[f"matmul_{index}"]})CUPTX", "matmul_{index}_kernel");""")
    for index, _ in enumerate(layer_norms):
        function_loads.append(f"""  CUfunction layernorm{index}_fn = load_function(
      R"CUPTX({ptx_by_key[f"layernorm_{index}"]})CUPTX", "layernorm_{index}_kernel");""")
    for index, _ in enumerate(softmaxes):
        function_loads.append(f"""  CUfunction softmax{index}_fn = load_function(
      R"CUPTX({ptx_by_key[f"softmax_{index}"]})CUPTX", "softmax_{index}_kernel");""")
    declarations = "\n".join(function_loads)
    check_functions = (
        " || ".join(
            [f"!linear{index}_fn" for index in range(len(linears))]
            + [f"!conv{index}_fn" for index in range(len(convs))]
            + [f"!relu{index}_fn" for index in range(len(relus))]
            + [f"!activation{index}_fn" for index in range(len(activations))]
            + [f"!copy{index}_fn" for index in range(len(copies))]
            + [f"!batchnorm{index}_fn" for index in range(len(batch_norms))]
            + [f"!maxpool{index}_fn" for index in range(len(max_pools))]
            + [f"!avgpool{index}_fn" for index in range(len(avg_pools))]
            + [f"!adaptive_avgpool{index}_fn" for index in range(len(adaptive_avg_pools))]
            + [f"!adaptive_maxpool{index}_fn" for index in range(len(adaptive_max_pools))]
            + [f"!binary{index}_fn" for index in range(len(binaries))]
            + [f"!cat{index}_fn" for index in range(len(cats))]
            + [f"!transpose{index}_fn" for index in range(len(transposes))]
            + [f"!matmul{index}_fn" for index in range(len(matmuls))]
            + [f"!layernorm{index}_fn" for index in range(len(layer_norms))]
            + [f"!softmax{index}_fn" for index in range(len(softmaxes))]
        )
        or "false"
    )

    path_decls = (
        "\n".join(
            f"  std::string weight{index}_path;\n  std::string bias{index}_path;"
            for index in range(len(weighted))
        )
        + "\n"
        + "\n".join(f"""  std::string bn_weight{index}_path;
  std::string bn_bias{index}_path;
  std::string bn_mean{index}_path;
  std::string bn_var{index}_path;""" for index in range(len(batch_norms)))
        + "\n"
        + "\n".join(f"""  std::string ln_weight{index}_path;
  std::string ln_bias{index}_path;""" for index in range(len(layer_norms)))
    )

    weight_arg_branches = []
    for index, _ in enumerate(weighted):
        weight_arg_branches.append(f"""    }} else if (arg == "--weight{index}" && i + 1 < argc) {{
      weight{index}_path = argv[++i];
    }} else if (arg == "--bias{index}" && i + 1 < argc) {{
      bias{index}_path = argv[++i];""")
    bn_arg_branches = []
    for index, _ in enumerate(batch_norms):
        bn_arg_branches.append(f"""    }} else if (arg == "--bn-weight{index}" && i + 1 < argc) {{
      bn_weight{index}_path = argv[++i];
    }} else if (arg == "--bn-bias{index}" && i + 1 < argc) {{
      bn_bias{index}_path = argv[++i];
    }} else if (arg == "--bn-mean{index}" && i + 1 < argc) {{
      bn_mean{index}_path = argv[++i];
    }} else if (arg == "--bn-var{index}" && i + 1 < argc) {{
      bn_var{index}_path = argv[++i];""")
    ln_arg_branches = []
    for index, _ in enumerate(layer_norms):
        ln_arg_branches.append(f"""    }} else if (arg == "--ln-weight{index}" && i + 1 < argc) {{
      ln_weight{index}_path = argv[++i];
    }} else if (arg == "--ln-bias{index}" && i + 1 < argc) {{
      ln_bias{index}_path = argv[++i];""")
    arg_parser = "\n".join(weight_arg_branches + bn_arg_branches + ln_arg_branches)

    weight_reads = []
    for index, op in enumerate(weighted):
        if op["kind"] == "linear":
            weight_count = op["out_features"] * op["in_features"]
            bias_count = op["out_features"]
        else:
            weight_count = op["out_channels"] * op["in_channels"] * op["kernel_h"] * op["kernel_w"]
            bias_count = op["out_channels"]
        weight_reads.append(f"""  const auto weight{index} = read_floats(weight{index}_path);
  const auto bias{index} = read_floats(bias{index}_path);
  if (static_cast<int>(weight{index}.size()) != {weight_count} ||
      static_cast<int>(bias{index}.size()) != {bias_count}) {{
    return 3;
  }}""")

    bn_reads = []
    for index, op in enumerate(batch_norms):
        count = op["channels"]
        bn_reads.append(f"""  const auto bn_weight{index} = read_floats(bn_weight{index}_path);
  const auto bn_bias{index} = read_floats(bn_bias{index}_path);
  const auto bn_mean{index} = read_floats(bn_mean{index}_path);
  const auto bn_var{index} = read_floats(bn_var{index}_path);
  if (static_cast<int>(bn_weight{index}.size()) != {count} ||
      static_cast<int>(bn_bias{index}.size()) != {count} ||
      static_cast<int>(bn_mean{index}.size()) != {count} ||
      static_cast<int>(bn_var{index}.size()) != {count}) {{
    return 3;
  }}""")

    ln_reads = []
    for index, op in enumerate(layer_norms):
        count = op["normalized_size"]
        ln_reads.append(f"""  const auto ln_weight{index} = read_floats(ln_weight{index}_path);
  const auto ln_bias{index} = read_floats(ln_bias{index}_path);
  if (static_cast<int>(ln_weight{index}.size()) != {count} ||
      static_cast<int>(ln_bias{index}.size()) != {count}) {{
    return 3;
  }}""")

    device_weight_allocs = "\n".join(
        f"""  CUdeviceptr d_weight{index} =
      allocate_and_copy(weight{index}.data(),
                        weight{index}.size() * sizeof(float));
  CUdeviceptr d_bias{index} =
      allocate_and_copy(bias{index}.data(), bias{index}.size() * sizeof(float));"""
        for index in range(len(weighted))
    )
    device_bn_allocs = "\n".join(
        f"""  CUdeviceptr d_bn_weight{index} =
      allocate_and_copy(bn_weight{index}.data(),
                        bn_weight{index}.size() * sizeof(float));
  CUdeviceptr d_bn_bias{index} =
      allocate_and_copy(bn_bias{index}.data(),
                        bn_bias{index}.size() * sizeof(float));
  CUdeviceptr d_bn_mean{index} =
      allocate_and_copy(bn_mean{index}.data(),
                        bn_mean{index}.size() * sizeof(float));
  CUdeviceptr d_bn_var{index} =
      allocate_and_copy(bn_var{index}.data(),
                        bn_var{index}.size() * sizeof(float));"""
        for index in range(len(batch_norms))
    )
    device_ln_allocs = "\n".join(
        f"""  CUdeviceptr d_ln_weight{index} =
      allocate_and_copy(ln_weight{index}.data(),
                        ln_weight{index}.size() * sizeof(float));
  CUdeviceptr d_ln_bias{index} =
      allocate_and_copy(ln_bias{index}.data(),
                        ln_bias{index}.size() * sizeof(float));"""
        for index in range(len(layer_norms))
    )
    alloc_guard_parts = ["!d_input"]
    alloc_guard_parts.extend(f"!d_weight{index}" for index in range(len(weighted)))
    alloc_guard_parts.extend(f"!d_bias{index}" for index in range(len(weighted)))
    alloc_guard_parts.extend(f"!d_bn_weight{index}" for index in range(len(batch_norms)))
    alloc_guard_parts.extend(f"!d_bn_bias{index}" for index in range(len(batch_norms)))
    alloc_guard_parts.extend(f"!d_bn_mean{index}" for index in range(len(batch_norms)))
    alloc_guard_parts.extend(f"!d_bn_var{index}" for index in range(len(batch_norms)))
    alloc_guard_parts.extend(f"!d_ln_weight{index}" for index in range(len(layer_norms)))
    alloc_guard_parts.extend(f"!d_ln_bias{index}" for index in range(len(layer_norms)))
    alloc_guard = " ||\n      ".join(alloc_guard_parts)

    buffer_allocs = []
    for op in program:
        var = _buffer_name(op["name"])
        buffer_allocs.append(f"""  CUdeviceptr {var} = 0;
  if (!check(cuMemAlloc(&{var}, {_numel(op["shape"])} * sizeof(float)))) {{
    return 5;
  }}""")

    launches = []
    for op in program:
        output_var = _buffer_name(op["name"])
        if op["kind"] == "binary":
            binary_index = _binary_index(binaries, op)
            left_name, right_name = op["inputs"]
            left_var = "d_input" if left_name == placeholder.name else _buffer_name(left_name)
            right_var = "d_input" if right_name == placeholder.name else _buffer_name(right_name)
            count = _numel(op["shape"])
            launches.append(f"""  if (!launch_binary(binary{binary_index}_fn, {left_var},
                       {right_var}, {output_var}, {count}, {args.block})) {{
    return 16;
  }}""")
            continue
        if op["kind"] == "cat":
            cat_index = _cat_index(cats, op)
            left_name, right_name = op["inputs"]
            left_var = "d_input" if left_name == placeholder.name else _buffer_name(left_name)
            right_var = "d_input" if right_name == placeholder.name else _buffer_name(right_name)
            count = _numel(op["shape"])
            launches.append(f"""  if (!launch_cat(cat{cat_index}_fn, {left_var},
                       {right_var}, {output_var}, {count}, {args.block})) {{
    return 20;
  }}""")
            continue
        if op["kind"] == "matmul":
            matmul_index = _matmul_index(matmuls, op)
            left_name, right_name = op["inputs"]
            left_var = "d_input" if left_name == placeholder.name else _buffer_name(left_name)
            right_var = "d_input" if right_name == placeholder.name else _buffer_name(right_name)
            m, n = op["shape"]
            launches.append(f"""  if (!launch_matmul(matmul{matmul_index}_fn, {left_var},
                      {right_var}, {output_var}, {m}, {n})) {{
    return 21;
  }}""")
            continue
        input_var = "d_input" if op["input"] == placeholder.name else _buffer_name(op["input"])
        if op["kind"] == "linear":
            linear_index = _linear_index(linears, op)
            weight_index = _weighted_index(weighted, op)
            launches.append(f"""  if (!launch_linear(linear{linear_index}_fn, {input_var},
                     d_weight{weight_index}, d_bias{weight_index}, {output_var},
                     {op["shape"][0]}, {op["out_features"]})) {{
    return 6;
  }}""")
        elif op["kind"] == "conv2d":
            conv_index = _conv_index(convs, op)
            weight_index = _weighted_index(weighted, op)
            n, c, h, w = op["input_shape"]
            k = op["out_channels"]
            r = op["kernel_h"]
            s = op["kernel_w"]
            stride = op["stride_h"]
            pad = op["pad_h"]
            out_h = op["shape"][2]
            out_w = op["shape"][3]
            launches.append(f"""  if (!launch_conv(conv{conv_index}_fn, {input_var},
                   d_weight{weight_index}, d_bias{weight_index}, {output_var},
                   {n}, {k}, {out_h}, {out_w})) {{
    return 8;
  }}""")
        elif op["kind"] == "batchnorm2d":
            batch_norm_index = _batch_norm_index(batch_norms, op)
            count = _numel(op["shape"])
            launches.append(f"""  if (!launch_batchnorm(batchnorm{batch_norm_index}_fn, {input_var},
                         d_bn_weight{batch_norm_index}, d_bn_bias{batch_norm_index},
                         d_bn_mean{batch_norm_index}, d_bn_var{batch_norm_index},
                         {output_var}, {count})) {{
    return 10;
  }}""")
        elif op["kind"] == "maxpool2d":
            max_pool_index = _max_pool_index(max_pools, op)
            n, c, h, w = op["input_shape"]
            out_h = op["shape"][2]
            out_w = op["shape"][3]
            launches.append(f"""  if (!launch_maxpool(maxpool{max_pool_index}_fn, {input_var},
                       {output_var}, {n * c}, {out_h}, {out_w})) {{
    return 11;
  }}""")
        elif op["kind"] == "avgpool2d":
            avg_pool_index = _avg_pool_index(avg_pools, op)
            n, c, h, w = op["input_shape"]
            out_h = op["shape"][2]
            out_w = op["shape"][3]
            launches.append(f"""  if (!launch_avgpool(avgpool{avg_pool_index}_fn, {input_var},
                       {output_var}, {n * c}, {out_h}, {out_w})) {{
    return 12;
  }}""")
        elif op["kind"] == "adaptive_avgpool2d":
            adaptive_pool_index = _adaptive_avg_pool_index(adaptive_avg_pools, op)
            batch_channels = _numel(op["shape"])
            launches.append(
                f"""  if (!launch_adaptive_avgpool(adaptive_avgpool{adaptive_pool_index}_fn,
                       {input_var}, {output_var}, {batch_channels})) {{
    return 13;
  }}"""
            )
        elif op["kind"] == "adaptive_maxpool2d":
            adaptive_pool_index = _adaptive_max_pool_index(adaptive_max_pools, op)
            batch_channels = _numel(op["shape"])
            launches.append(
                f"""  if (!launch_adaptive_maxpool(adaptive_maxpool{adaptive_pool_index}_fn,
                       {input_var}, {output_var}, {batch_channels})) {{
    return 19;
  }}"""
            )
        elif op["kind"] == "relu":
            index = _relu_index(relus, op)
            count = _numel(op["shape"])
            launches.append(f"""  if (!launch_elementwise(relu{index}_fn, {input_var},
                          {output_var}, {count}, {args.block})) {{
    return 7;
  }}""")
        elif op["kind"] == "activation":
            index = _activation_index(activations, op)
            count = _numel(op["shape"])
            launches.append(f"""  if (!launch_elementwise(activation{index}_fn, {input_var},
                          {output_var}, {count}, {args.block})) {{
    return 14;
  }}""")
        elif op["kind"] == "copy":
            index = _copy_index(copies, op)
            count = _numel(op["shape"])
            launches.append(f"""  if (!launch_elementwise(copy{index}_fn, {input_var},
                          {output_var}, {count}, {args.block})) {{
    return 15;
  }}""")
        elif op["kind"] == "layernorm":
            layer_norm_index = _layer_norm_index(layer_norms, op)
            count = _numel(op["shape"])
            launches.append(f"""  if (!launch_layernorm(layernorm{layer_norm_index}_fn, {input_var},
                         d_ln_weight{layer_norm_index}, d_ln_bias{layer_norm_index},
                         {output_var}, {count})) {{
    return 17;
  }}""")
        elif op["kind"] == "transpose":
            transpose_index = _transpose_index(transposes, op)
            count = _numel(op["shape"])
            launches.append(f"""  if (!launch_transpose(transpose{transpose_index}_fn, {input_var},
                         {output_var}, {count}, {args.block})) {{
    return 22;
  }}""")
        elif op["kind"] == "softmax":
            softmax_index = _softmax_index(softmaxes, op)
            count = _numel(op["shape"])
            launches.append(f"""  if (!launch_softmax(softmax{softmax_index}_fn, {input_var},
                       {output_var}, {count})) {{
    return 18;
  }}""")
        else:
            raise RuntimeError(f"unsupported program operation: {op['kind']}")

    final_var = _buffer_name(final_node.name)
    final_count = _numel(shapes[final_node.name])
    placeholder_count = _numel(shapes[placeholder.name])

    launch_helpers = []
    if linears:
        launch_helpers.append(
            """bool launch_linear(CUfunction function, CUdeviceptr x, CUdeviceptr w,
                   CUdeviceptr b, CUdeviceptr out, int m, int n) {
  CUdeviceptr scratch = 0;
  CUdeviceptr profile_scratch = 0;
  void *params[] = {&x, &w, &b, &out, &m, &n, &scratch, &profile_scratch};
  return check(cuLaunchKernel(function, static_cast<unsigned int>(m),
                              static_cast<unsigned int>(n), 1, 32, 1, 1, 0,
                              nullptr, params, nullptr));
}"""
        )
    if convs:
        launch_helpers.append("""bool launch_conv(CUfunction function, CUdeviceptr x, CUdeviceptr w,
                 CUdeviceptr b, CUdeviceptr out, int n, int k, int out_h,
                 int out_w) {
  CUdeviceptr scratch = 0;
  CUdeviceptr profile_scratch = 0;
  void *params[] = {&x, &w, &b, &out, &scratch, &profile_scratch};
  return check(cuLaunchKernel(function, static_cast<unsigned int>(n * k),
                              static_cast<unsigned int>(out_h),
                              static_cast<unsigned int>(out_w), 32, 1, 1, 0,
                              nullptr, params, nullptr));
}""")
    if relus or activations or copies:
        launch_helpers.append("""bool launch_elementwise(CUfunction function, CUdeviceptr x,
                        CUdeviceptr out, int count, int block) {
  CUdeviceptr scratch = 0;
  CUdeviceptr profile_scratch = 0;
  void *params[] = {&x, &out, &count, &scratch, &profile_scratch};
  const unsigned int grid =
      (static_cast<unsigned int>(count) + static_cast<unsigned int>(block) - 1) /
      static_cast<unsigned int>(block);
  return check(cuLaunchKernel(function, grid, 1, 1,
                              static_cast<unsigned int>(block), 1, 1, 0,
                              nullptr, params, nullptr));
}""")
    if batch_norms:
        launch_helpers.append("""bool launch_batchnorm(CUfunction function, CUdeviceptr x,
                       CUdeviceptr weight, CUdeviceptr bias, CUdeviceptr mean,
                       CUdeviceptr var, CUdeviceptr out, int count) {
  CUdeviceptr scratch = 0;
  CUdeviceptr profile_scratch = 0;
  void *params[] = {&x, &weight, &bias, &mean, &var, &out, &scratch, &profile_scratch};
  return check(cuLaunchKernel(function, static_cast<unsigned int>(count), 1, 1,
                              32, 1, 1, 0, nullptr, params, nullptr));
}""")
    if max_pools:
        launch_helpers.append("""bool launch_maxpool(CUfunction function, CUdeviceptr x,
                    CUdeviceptr out, int batch_channels, int out_h,
                    int out_w) {
  CUdeviceptr scratch = 0;
  CUdeviceptr profile_scratch = 0;
  void *params[] = {&x, &out, &scratch, &profile_scratch};
  return check(cuLaunchKernel(function,
                              static_cast<unsigned int>(batch_channels),
                              static_cast<unsigned int>(out_h),
                              static_cast<unsigned int>(out_w), 32, 1, 1, 0,
                              nullptr, params, nullptr));
}""")
    if avg_pools:
        launch_helpers.append("""bool launch_avgpool(CUfunction function, CUdeviceptr x,
                   CUdeviceptr out, int batch_channels, int out_h,
                   int out_w) {
  CUdeviceptr scratch = 0;
  CUdeviceptr profile_scratch = 0;
  void *params[] = {&x, &out, &scratch, &profile_scratch};
  return check(cuLaunchKernel(function,
                              static_cast<unsigned int>(batch_channels),
                              static_cast<unsigned int>(out_h),
                              static_cast<unsigned int>(out_w), 32, 1, 1, 0,
                              nullptr, params, nullptr));
}""")
    if adaptive_avg_pools:
        launch_helpers.append("""bool launch_adaptive_avgpool(CUfunction function, CUdeviceptr x,
                              CUdeviceptr out, int batch_channels) {
  CUdeviceptr scratch = 0;
  CUdeviceptr profile_scratch = 0;
  void *params[] = {&x, &out, &scratch, &profile_scratch};
  return check(cuLaunchKernel(function,
                              static_cast<unsigned int>(batch_channels), 1, 1,
                              32, 1, 1, 0, nullptr, params, nullptr));
}""")
    if adaptive_max_pools:
        launch_helpers.append("""bool launch_adaptive_maxpool(CUfunction function, CUdeviceptr x,
                              CUdeviceptr out, int batch_channels) {
  CUdeviceptr scratch = 0;
  CUdeviceptr profile_scratch = 0;
  void *params[] = {&x, &out, &scratch, &profile_scratch};
  return check(cuLaunchKernel(function,
                              static_cast<unsigned int>(batch_channels), 1, 1,
                              32, 1, 1, 0, nullptr, params, nullptr));
}""")
    if binaries:
        launch_helpers.append("""bool launch_binary(CUfunction function, CUdeviceptr x,
                   CUdeviceptr y, CUdeviceptr out, int count, int block) {
  CUdeviceptr scratch = 0;
  CUdeviceptr profile_scratch = 0;
  void *params[] = {&x, &y, &out, &count, &scratch, &profile_scratch};
  const unsigned int grid =
      (static_cast<unsigned int>(count) + static_cast<unsigned int>(block) - 1) /
      static_cast<unsigned int>(block);
  return check(cuLaunchKernel(function, grid, 1, 1,
                              static_cast<unsigned int>(block), 1, 1, 0,
                              nullptr, params, nullptr));
}""")
    if cats:
        launch_helpers.append("""bool launch_cat(CUfunction function, CUdeviceptr x,
                 CUdeviceptr y, CUdeviceptr out, int count, int block) {
  CUdeviceptr scratch = 0;
  CUdeviceptr profile_scratch = 0;
  void *params[] = {&x, &y, &out, &count, &scratch, &profile_scratch};
  const unsigned int grid =
      (static_cast<unsigned int>(count) + static_cast<unsigned int>(block) - 1) /
      static_cast<unsigned int>(block);
  return check(cuLaunchKernel(function, grid, 1, 1,
                              static_cast<unsigned int>(block), 1, 1, 0,
                              nullptr, params, nullptr));
}""")
    if transposes:
        launch_helpers.append("""bool launch_transpose(CUfunction function, CUdeviceptr x,
                       CUdeviceptr out, int count, int block) {
  CUdeviceptr scratch = 0;
  CUdeviceptr profile_scratch = 0;
  void *params[] = {&x, &out, &scratch, &profile_scratch};
  const unsigned int grid =
      (static_cast<unsigned int>(count) + static_cast<unsigned int>(block) - 1) /
      static_cast<unsigned int>(block);
  return check(cuLaunchKernel(function, grid, 1, 1,
                              static_cast<unsigned int>(block), 1, 1, 0,
                              nullptr, params, nullptr));
}""")
    if matmuls:
        launch_helpers.append("""bool launch_matmul(CUfunction function, CUdeviceptr x,
                   CUdeviceptr y, CUdeviceptr out, int m, int n) {
  CUdeviceptr scratch = 0;
  CUdeviceptr profile_scratch = 0;
  void *params[] = {&x, &y, &out, &scratch, &profile_scratch};
  return check(cuLaunchKernel(function, static_cast<unsigned int>(m),
                              static_cast<unsigned int>(n), 1, 32, 1, 1, 0,
                              nullptr, params, nullptr));
}""")
    if layer_norms:
        launch_helpers.append("""bool launch_layernorm(CUfunction function, CUdeviceptr x,
                       CUdeviceptr weight, CUdeviceptr bias, CUdeviceptr out,
                       int count) {
  CUdeviceptr scratch = 0;
  CUdeviceptr profile_scratch = 0;
  void *params[] = {&x, &weight, &bias, &out, &scratch, &profile_scratch};
  return check(cuLaunchKernel(function, static_cast<unsigned int>(count), 1, 1,
                              32, 1, 1, 0, nullptr, params, nullptr));
}""")
    if softmaxes:
        launch_helpers.append("""bool launch_softmax(CUfunction function, CUdeviceptr x,
                    CUdeviceptr out, int count) {
  CUdeviceptr scratch = 0;
  CUdeviceptr profile_scratch = 0;
  void *params[] = {&x, &out, &scratch, &profile_scratch};
  return check(cuLaunchKernel(function, static_cast<unsigned int>(count), 1, 1,
                              32, 1, 1, 0, nullptr, params, nullptr));
}""")
    launch_helpers_text = "\n".join(launch_helpers)

    progress_launches = []
    for index, launch in enumerate(launches):
        node_name = program[index]["name"]
        progress_launches.append(
            launch
            + f"""\n  std::fprintf(stderr, "[model] completed %d/%d nodes (%.1f%%): {node_name}\\n",
               ++completed_nodes, total_nodes,
               100.0 * static_cast<double>(completed_nodes) / total_nodes);"""
        )

    source = runner_prelude() + f"""
{launch_helpers_text}

}} // namespace

int main(int argc, char **argv) {{
  std::string input_path;
  std::string output_path;
{path_decls}

  for (int i = 1; i < argc; ++i) {{
    const std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {{
      input_path = argv[++i];
    }} else if (arg == "--output" && i + 1 < argc) {{
      output_path = argv[++i];
{arg_parser}
    }}
  }}

  if (input_path.empty() || output_path.empty()) {{
    return 1;
  }}
  if (!initialize_cuda()) {{
    return 2;
  }}

  const auto input = read_floats(input_path);
  if (static_cast<int>(input.size()) != {placeholder_count}) {{
    return 3;
  }}

{chr(10).join(weight_reads)}
{chr(10).join(bn_reads)}
{chr(10).join(ln_reads)}

{declarations}
  if ({check_functions}) {{
    return 4;
  }}

  CUdeviceptr d_input = allocate_and_copy(input.data(), input.size() * sizeof(float));
{device_weight_allocs}
{device_bn_allocs}
{device_ln_allocs}
  if ({alloc_guard}) {{
    return 5;
  }}

{chr(10).join(buffer_allocs)}

  const int total_nodes = {len(program)};
  int completed_nodes = 0;

{chr(10).join(progress_launches)}

  std::vector<float> result({final_count});
  if (!check(cuMemcpyDtoH(result.data(), {final_var},
                          result.size() * sizeof(float)))) {{
    return 9;
  }}
  write_floats(output_path, result);
  return 0;
}}
"""
    return source
