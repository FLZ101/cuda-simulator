#!/usr/bin/env python3

import argparse
import math
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
    runner = args.runner

    gpu_env = os.environ.copy()
    sim_env = env

    def run_both(mode, make_args):
        gpu_out = os.path.join(tmp, f"{mode}_gpu.bin")
        sim_out = os.path.join(tmp, f"{mode}_sim.bin")
        subprocess.run(
            [runner, mode, *make_args(gpu_out)],
            check=True,
            env=gpu_env,
        )
        subprocess.run(
            [runner, mode, *make_args(sim_out)],
            check=True,
            env=sim_env,
        )
        gpu_values = read_floats(gpu_out)
        sim_values = read_floats(sim_out)
        assert len(gpu_values) == len(sim_values), f"{mode} length mismatch"
        for a, b in zip(gpu_values, sim_values):
            if abs(a - b) > 1e-4 + 1e-5 * abs(b):
                raise AssertionError(f"{mode} value mismatch")

    with tempfile.TemporaryDirectory() as tmp:
        n = 128
        x = [float(i) for i in range(n)]
        y = [float(i * 2) for i in range(n)]
        alpha = 0.5
        x_path = os.path.join(tmp, "x.bin")
        y_path = os.path.join(tmp, "y.bin")
        write_floats(x_path, x)
        write_floats(y_path, y)

        run_both(
            "saxpy",
            lambda out: [x_path, y_path, out, str(alpha), str(n)],
        )
        run_both("copy", lambda out: [x_path, out, str(n)])
        run_both("relu", lambda out: [x_path, out, str(n)])
        run_both("gelu", lambda out: [x_path, out, str(n)])

    print("demo kernels: PASS")


if __name__ == "__main__":
    main()
