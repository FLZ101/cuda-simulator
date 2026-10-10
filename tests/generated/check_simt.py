#!/usr/bin/env python3

import pathlib
import re
import sys

BANNED_PATTERN = re.compile(r"\b(mma|wmma|ldmatrix|stmatrix|wgmma|tcgen05)\b")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: check_simt.py PTX [PTX ...]")
        return 2

    failed = False
    for raw_path in sys.argv[1:]:
        path = pathlib.Path(raw_path)
        text = path.read_text(errors="replace")
        matches = BANNED_PATTERN.findall(text)
        if matches:
            failed = True
            unique = sorted(set(matches))
            print(f"{path}: FAIL tensor-core instructions: {unique}")
        else:
            print(f"{path}: PASS")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
