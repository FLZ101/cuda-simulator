#!/usr/bin/env python3

import argparse
import os
import struct
import subprocess
import tempfile


def write_floats(path, values):
    with open(path, "wb") as f:
        f.write(b"".join(struct.pack("<f", value) for value in values))


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
    args = parser.parse_args()

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = args.core_dir
    env["LD_PRELOAD"] = f"{args.cudart}:{args.cuda}"

    with tempfile.TemporaryDirectory() as tmp:
        n = 128
        x = [float(i) for i in range(n)]
        y = [float(i * 2) for i in range(n)]
        alpha = 0.5
        expected_saxpy = [alpha * a + b for a, b in zip(x, y)]
        x_path = os.path.join(tmp, "x.bin")
        y_path = os.path.join(tmp, "y.bin")
        saxpy_out = os.path.join(tmp, "saxpy.bin")
        write_floats(x_path, x)
        write_floats(y_path, y)
        subprocess.run(
            [args.runner, "saxpy", x_path, y_path, saxpy_out, str(alpha), str(n)],
            check=True,
            env=env,
        )
        actual_saxpy = read_floats(saxpy_out)
        assert actual_saxpy == expected_saxpy, "saxpy mismatch"

        copy_out = os.path.join(tmp, "copy.bin")
        subprocess.run(
            [args.runner, "copy", x_path, copy_out, str(n)],
            check=True,
            env=env,
        )
        actual_copy = read_floats(copy_out)
        assert actual_copy == x, "copy mismatch"

    print("demo kernels: PASS")


if __name__ == "__main__":
    main()
