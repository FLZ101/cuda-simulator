#!/usr/bin/env python3

import argparse
import importlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-cpp", required=True)
    known, _ = parser.parse_known_args()

    try:
        spec = importlib.import_module(f"models.specs.{known.spec}")
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"unknown model spec: {known.spec}") from exc

    spec.add_arguments(parser)
    parsed = parser.parse_args()
    output_path = pathlib.Path(parsed.output_cpp)
    spec.generate_native_runner(parsed, output_path)
    print(f"generated native runner: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
