#!/usr/bin/env python3

import argparse
import os
import pathlib
import struct
import subprocess
import tempfile

from compare import compare_float_lists
from tolerance import TOLERANCES


def write_floats(path, values):
    with open(path, "wb") as f:
        f.write(b"".join(struct.pack("<f", value) for value in values))


def run_vector_add(runner, cudart, cuda, core_dir, tmpdir):
    n = 256
    a = [float(i) for i in range(n)]
    b = [float(2 * i + 1) for i in range(n)]
    expected = [x + y for x, y in zip(a, b)]
    a_path = os.path.join(tmpdir, "a.bin")
    b_path = os.path.join(tmpdir, "b.bin")
    out_path = os.path.join(tmpdir, "out.bin")
    write_floats(a_path, a)
    write_floats(b_path, b)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = core_dir
    env["LD_PRELOAD"] = f"{cudart}:{cuda}"
    subprocess.run([runner, a_path, b_path, out_path, str(n)], check=True, env=env)

    with open(out_path, "rb") as f:
        raw = f.read()
    actual = list(struct.unpack("<" + "f" * (len(raw) // 4), raw))
    rtol, atol = TOLERANCES["fp32"]
    ok, message = compare_float_lists(actual, expected, rtol, atol)
    if not ok:
        raise SystemExit(f"vector_add mismatch: {message}")
    print(f"vector_add: PASS ({message})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True)
    parser.add_argument("--cudart", required=True)
    parser.add_argument("--cuda", required=True)
    parser.add_argument("--core-dir", required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        run_vector_add(args.runner, args.cudart, args.cuda, args.core_dir, tmpdir)


if __name__ == "__main__":
    main()
