#!/usr/bin/env python3

import argparse
import pathlib
import struct

import torch


def read_floats(path: pathlib.Path) -> list[float]:
    raw = path.read_bytes()
    count = len(raw) // 4
    if count * 4 != len(raw):
        raise ValueError(f"float binary file is not 4-byte aligned: {path}")
    return list(struct.unpack("<" + "f" * count, raw))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator-output", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    reference = torch.load(args.reference, weights_only=False)
    expected = reference["output"].reshape(-1).to(torch.float32)
    actual_values = read_floats(pathlib.Path(args.simulator_output))
    actual = torch.tensor(actual_values, dtype=torch.float32)

    if actual.numel() != expected.numel():
        raise AssertionError(f"output length mismatch: {actual.numel()} != {expected.numel()}")

    if not torch.allclose(actual, expected, rtol=args.rtol, atol=args.atol):
        diff = (actual - expected).abs()
        raise AssertionError(
            "PyTorch CPU reference mismatch: "
            f"max_abs={diff.max().item():.6e} "
            f"rtol={args.rtol:.1e} atol={args.atol:.1e}"
        )

    print(f"PyTorch CPU reference matches ({actual.numel()} floats)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
