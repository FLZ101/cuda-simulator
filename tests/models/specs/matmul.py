#!/usr/bin/env python3

import argparse
import pathlib

import torch
import triton
import triton.language as tl

from ._common import Arg, Input, Kernel, Runner, compile_kernel, write_floats


@triton.jit
def matmul_scalar_kernel(a_ptr, b_ptr, out_ptr, m, n, k: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    acc = 0.0
    for kk in tl.range(0, k):
        a = tl.load(a_ptr + row * k + kk)
        b = tl.load(b_ptr + kk * n + col)
        acc += a * b
    tl.store(out_ptr + row * n + col, acc)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--k", type=int, default=8)


def _generate_ptx(m: int, n: int, k: int) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-capable PyTorch is required to generate Triton PTX")

    a = torch.randn(m, k, device="cuda", dtype=torch.float32)
    b = torch.randn(k, n, device="cuda", dtype=torch.float32)
    out = torch.empty(m, n, device="cuda", dtype=torch.float32)
    return compile_kernel(
        matmul_scalar_kernel,
        (m, n),
        (a, b, out, m, n),
        {"k": k, "num_warps": 1},
    )


def run_cpu(args: argparse.Namespace, output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1234)
    a = torch.randn(args.m, args.k, dtype=torch.float32)
    b = torch.randn(args.k, args.n, dtype=torch.float32)
    with torch.no_grad():
        output = torch.matmul(a, b)

    torch.save(
        {
            "a": a,
            "b": b,
            "output": output,
            "shape": tuple(output.shape),
            "dtype": str(output.dtype),
        },
        output_dir / "reference.pt",
    )
    write_floats(output_dir / "a.bin", a)
    write_floats(output_dir / "b.bin", b)
    write_floats(output_dir / "cpu_output.bin", output)


def native_arguments(args: argparse.Namespace, output_dir: pathlib.Path) -> list[str]:
    return [
        "--input-a",
        str(output_dir / "a.bin"),
        "--input-b",
        str(output_dir / "b.bin"),
        "--output",
        str(output_dir / "gpu_output.bin"),
        "--m",
        str(args.m),
        "--n",
        str(args.n),
        "--k",
        str(args.k),
    ]


def build_runner(args: argparse.Namespace) -> Runner:
    ptx = _generate_ptx(args.m, args.n, args.k)
    return Runner(
        args=[
            Arg("--input-a", "input_a", "string"),
            Arg("--input-b", "input_b", "string"),
            Arg("--m", "m", "int"),
            Arg("--n", "n", "int"),
            Arg("--k", "k", "int"),
        ],
        inputs=[
            Input("a", "input_a", "m * k"),
            Input("b", "input_b", "k * n"),
        ],
        output="d_out",
        output_count="m * n",
        kernels=[
            Kernel(
                "matmul_scalar_kernel",
                ptx,
                ("static_cast<unsigned int>(m)", "static_cast<unsigned int>(n)", "1"),
                "32",
                ["&d_a", "&d_b", "&d_out", "&m", "&n"],
            ),
        ],
    )
