import math
import struct
import sys


def read_floats(path):
    values = []
    with open(path, "rb") as f:
        while True:
            raw = f.read(4)
            if not raw:
                break
            values.append(struct.unpack("<f", raw)[0])
    return values


def compare_float_lists(actual, expected, rtol, atol):
    if len(actual) != len(expected):
        return False, "length mismatch"
    max_abs = 0.0
    max_rel = 0.0
    for a, e in zip(actual, expected):
        abs_err = abs(a - e)
        max_abs = max(max_abs, abs_err)
        if abs(e) > 0.0:
            max_rel = max(max_rel, abs_err / abs(e))
        if abs_err > atol + rtol * abs(e):
            return False, f"max_abs={max_abs} max_rel={max_rel}"
    return True, f"max_abs={max_abs} max_rel={max_rel}"


def main():
    if len(sys.argv) != 5:
        return 2
    actual = read_floats(sys.argv[1])
    expected = read_floats(sys.argv[2])
    ok, message = compare_float_lists(actual, expected, float(sys.argv[3]), float(sys.argv[4]))
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
