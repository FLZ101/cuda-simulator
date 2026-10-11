#!/usr/bin/env python3

import argparse
import importlib
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--mode", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--runner", required=True)
    parser.add_argument("--cudart", required=True)
    parser.add_argument("--cuda", required=True)
    parser.add_argument("--core-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--compare", action="store_true")
    known, _ = parser.parse_known_args()

    try:
        spec = importlib.import_module(f"models.specs.{known.spec}")
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"unknown model spec: {known.spec}") from exc

    spec.add_arguments(parser)
    parsed = parser.parse_args()
    output_dir = pathlib.Path(parsed.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if parsed.mode == "cpu":
        spec.run_cpu(parsed, output_dir)
        print(f"{parsed.spec} CPU mode wrote reference to {output_dir}")
        return 0

    env = os.environ.copy()
    if parsed.simulate:
        env["LD_LIBRARY_PATH"] = parsed.core_dir
        env["LD_PRELOAD"] = f"{parsed.cudart}:{parsed.cuda}"

    command = [parsed.runner, *spec.native_arguments(parsed, output_dir)]
    subprocess.run(command, check=True, env=env)

    if parsed.compare:
        subprocess.run(
            [
                sys.executable,
                str(pathlib.Path(__file__).resolve().parent / "compare_torch_reference.py"),
                "--simulator-output",
                str(output_dir / "gpu_output.bin"),
                "--reference",
                str(output_dir / "reference.pt"),
            ],
            check=True,
        )

    mode_name = "simulator" if parsed.simulate else "gpu"
    print(f"{parsed.spec} {mode_name} mode passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
