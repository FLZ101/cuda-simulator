#!/usr/bin/env python3

import pathlib
import sys

TARGET_OPERATORS = [
    "elementwise",
    "conv2d",
    "gemm",
    "reduction",
    "softmax",
    "layer_norm",
    "attention",
    "embedding",
    "transpose",
    "cast",
]


def main() -> int:
    ptx_files = []
    if len(sys.argv) > 1:
        for raw_path in sys.argv[1:]:
            ptx_files.append(pathlib.Path(raw_path))

    print("target operator set:")
    for name in TARGET_OPERATORS:
        print(f"  {name}")
    print(f"generated PTX files known: {len(ptx_files)}")
    for path in ptx_files:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
