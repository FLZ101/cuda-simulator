#!/usr/bin/env python3

import argparse
import pathlib

import torch
import triton
import triton.language as tl

from ._common import Arg, Input, Kernel, Runner, compile_kernel, write_floats


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--block", type=int, default=32)


def _generate_ptx(n: int, block: int) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-capable PyTorch is required to generate Triton PTX")

    x = torch.empty(block, device="cuda", dtype=torch.float32)
    y = torch.empty(block, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    return compile_kernel(
        add_kernel,
        (1,),
        (x, y, out, block),
        {"BLOCK": block, "num_warps": 1},
    )


def run_cpu(args: argparse.Namespace, output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1234)
    a = torch.randn(args.n, dtype=torch.float32)
    b = torch.randn(args.n, dtype=torch.float32)
    with torch.no_grad():
        output = a + b

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
        "--n",
        str(args.n),
    ]


def build_runner(args: argparse.Namespace) -> Runner:
    ptx = _generate_ptx(args.n, args.block)
    return Runner(
        args=[
            Arg("--input-a", "input_a", "string"),
            Arg("--input-b", "input_b", "string"),
            Arg("--n", "n", "int"),
        ],
        inputs=[
            Input("a", "input_a", "n"),
            Input("b", "input_b", "n"),
        ],
        output="d_out",
        output_count="n",
        kernels=[
            Kernel(
                "add_kernel",
                ptx,
                (f"(static_cast<unsigned int>(n) + {args.block} - 1) / {args.block}", "1", "1"),
                f"{args.block}",
                ["&d_a", "&d_b", "&d_out", "&n"],
            ),
        ],
    )
