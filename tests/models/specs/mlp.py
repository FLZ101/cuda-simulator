#!/usr/bin/env python3

import argparse
import pathlib

import torch
import triton
import triton.language as tl

from ._common import Arg, Input, Kernel, Local, Runner, compile_kernel, write_floats


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


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--in-features", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--out-features", type=int, default=8)
    parser.add_argument("--block", type=int, default=32)


def run_cpu(args: argparse.Namespace, output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1234)
    model = torch.nn.Sequential(
        torch.nn.Linear(args.in_features, args.hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden, args.out_features),
    )
    model.eval()
    x = torch.randn(args.batch, args.in_features, dtype=torch.float32)
    with torch.no_grad():
        output = model(x)

    linear0 = model[0]
    linear1 = model[2]
    torch.save(
        {
            "x": x,
            "output": output,
            "shape": tuple(output.shape),
            "dtype": str(output.dtype),
        },
        output_dir / "reference.pt",
    )
    write_floats(output_dir / "input.bin", x)
    write_floats(output_dir / "weight0.bin", linear0.weight)
    write_floats(output_dir / "bias0.bin", linear0.bias)
    write_floats(output_dir / "weight1.bin", linear1.weight)
    write_floats(output_dir / "bias1.bin", linear1.bias)
    write_floats(output_dir / "cpu_output.bin", output)


def native_arguments(args: argparse.Namespace, output_dir: pathlib.Path) -> list[str]:
    return [
        "--input",
        str(output_dir / "input.bin"),
        "--weight0",
        str(output_dir / "weight0.bin"),
        "--bias0",
        str(output_dir / "bias0.bin"),
        "--weight1",
        str(output_dir / "weight1.bin"),
        "--bias1",
        str(output_dir / "bias1.bin"),
        "--output",
        str(output_dir / "gpu_output.bin"),
        "--batch",
        str(args.batch),
        "--in-features",
        str(args.in_features),
        "--hidden",
        str(args.hidden),
        "--out-features",
        str(args.out_features),
    ]


def build_runner(args: argparse.Namespace) -> Runner:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-capable PyTorch is required to generate Triton PTX")

    batch = args.batch
    in_features = args.in_features
    hidden = args.hidden
    out_features = args.out_features

    x = torch.randn(batch, in_features, device="cuda", dtype=torch.float32)
    w0 = torch.randn(hidden, in_features, device="cuda", dtype=torch.float32)
    b0 = torch.randn(hidden, device="cuda", dtype=torch.float32)
    mid = torch.empty(batch, hidden, device="cuda", dtype=torch.float32)
    linear0_ptx = compile_kernel(
        linear_kernel,
        (batch, hidden),
        (x, w0, b0, mid, batch, hidden),
        {"k": in_features, "num_warps": 1},
    ).replace("linear_kernel", "linear0_kernel")

    w1 = torch.randn(out_features, hidden, device="cuda", dtype=torch.float32)
    b1 = torch.randn(out_features, device="cuda", dtype=torch.float32)
    out0 = torch.empty(batch, out_features, device="cuda", dtype=torch.float32)
    linear1_ptx = compile_kernel(
        linear_kernel,
        (batch, out_features),
        (mid, w1, b1, out0, batch, out_features),
        {"k": hidden, "num_warps": 1},
    ).replace("linear_kernel", "linear1_kernel")

    act_in = torch.empty_like(mid)
    act_out = torch.empty_like(mid)
    relu_ptx = compile_kernel(
        relu_kernel,
        (1,),
        (act_in, act_out, mid.numel()),
        {"BLOCK": args.block, "num_warps": 1},
    )

    return Runner(
        args=[
            Arg("--input", "input_path", "string"),
            Arg("--weight0", "weight0_path", "string"),
            Arg("--bias0", "bias0_path", "string"),
            Arg("--weight1", "weight1_path", "string"),
            Arg("--bias1", "bias1_path", "string"),
            Arg("--batch", "batch", "int"),
            Arg("--in-features", "in_features", "int"),
            Arg("--hidden", "hidden", "int"),
            Arg("--out-features", "out_features", "int"),
        ],
        locals=[Local("int", "act_count", "batch * hidden")],
        inputs=[
            Input("input", "input_path", "batch * in_features"),
            Input("weight0", "weight0_path", "hidden * in_features"),
            Input("bias0", "bias0_path", "hidden"),
            Input("weight1", "weight1_path", "out_features * hidden"),
            Input("bias1", "bias1_path", "out_features"),
        ],
        intermediates=[
            ("d_mid", "batch * hidden"),
            ("d_act", "batch * hidden"),
        ],
        output="d_out",
        output_count="batch * out_features",
        kernels=[
            Kernel(
                "linear0_kernel",
                linear0_ptx,
                ("static_cast<unsigned int>(batch)", "static_cast<unsigned int>(hidden)", "1"),
                "32",
                ["&d_input", "&d_weight0", "&d_bias0", "&d_mid", "&batch", "&hidden"],
            ),
            Kernel(
                "relu_kernel",
                relu_ptx,
                (
                    f"(static_cast<unsigned int>(act_count) + {args.block} - 1) / {args.block}",
                    "1",
                    "1",
                ),
                f"{args.block}",
                ["&d_mid", "&d_act", "&act_count"],
            ),
            Kernel(
                "linear1_kernel",
                linear1_ptx,
                (
                    "static_cast<unsigned int>(batch)",
                    "static_cast<unsigned int>(out_features)",
                    "1",
                ),
                "32",
                ["&d_act", "&d_weight1", "&d_bias1", "&d_out", "&batch", "&out_features"],
            ),
        ],
    )
