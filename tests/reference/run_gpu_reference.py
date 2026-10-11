#!/usr/bin/env python3

import argparse
import math
import os
import struct
import subprocess
import tempfile

SUPPORTED_OPERATORS = {
    "bias_add",
    "batch_norm",
    "max_pool2d",
    "avg_pool2d",
    "layer_norm",
    "embedding_lookup",
    "transpose",
    "matmul_naive",
    "matmul_tiled",
    "softmax_row",
    "reduce_sum",
    "conv2d_naive",
    "conv2d_im2col",
    "conv2d_shared",
    "attention_scores",
    "gelu",
    "copy",
    "saxpy",
    "relu",
    "elementwise_add",
    "elementwise_mul",
    "scale",
    "mean",
    "dropout",
    "causal_mask",
    "residual_add",
    "rms_norm",
}


def write_floats(path, values):
    with open(path, "wb") as f:
        f.write(b"".join(struct.pack("<f", v) for v in values))


def write_ints(path, values):
    with open(path, "wb") as f:
        f.write(b"".join(struct.pack("<i", v) for v in values))


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

    gpu_env = os.environ.copy()
    sim_env = os.environ.copy()
    sim_env["LD_LIBRARY_PATH"] = args.core_dir
    sim_env["LD_PRELOAD"] = f"{args.cudart}:{args.cuda}"

    with tempfile.TemporaryDirectory() as tmp:

        def run_pair(mode, make_args):
            if mode not in SUPPORTED_OPERATORS:
                print(f"{mode}: skip (not enabled yet)")
                return
            gpu_out = os.path.join(tmp, f"{mode}_gpu.bin")
            sim_out = os.path.join(tmp, f"{mode}_sim.bin")
            subprocess.run([args.runner, mode, *make_args(gpu_out)], check=True, env=gpu_env)
            subprocess.run([args.runner, mode, *make_args(sim_out)], check=True, env=sim_env)
            gpu = read_floats(gpu_out)
            sim = read_floats(sim_out)
            assert len(gpu) == len(sim), f"{mode} length mismatch"
            for a, b in zip(gpu, sim):
                if abs(a - b) > 1e-4 + 1e-5 * abs(b):
                    raise AssertionError(f"{mode} value mismatch")

        n = 128
        x = [float(i) for i in range(n)]
        x_path = os.path.join(tmp, "x.bin")
        write_floats(x_path, x)
        y = [float(i * 2) for i in range(n)]
        y_path = os.path.join(tmp, "y.bin")
        write_floats(y_path, y)
        mask = [1.0 if i % 2 == 0 else 0.5 for i in range(n)]
        mask_path = os.path.join(tmp, "mask.bin")
        write_floats(mask_path, mask)
        causal = [float(i) for i in range(16)]
        causal_path = os.path.join(tmp, "causal.bin")
        write_floats(causal_path, causal)
        run_pair("gelu", lambda out: [x_path, str(n), out])
        run_pair("copy", lambda out: [x_path, str(n), out])
        run_pair("relu", lambda out: [x_path, str(n), out])
        run_pair(
            "saxpy",
            lambda out: [x_path, y_path, "0.5", str(n), out],
        )
        run_pair(
            "elementwise_add",
            lambda out: [x_path, y_path, str(n), out],
        )
        run_pair(
            "elementwise_mul",
            lambda out: [x_path, y_path, str(n), out],
        )
        run_pair("scale", lambda out: [x_path, "0.5", str(n), out])
        run_pair("mean", lambda out: [x_path, str(n), out])
        run_pair("dropout", lambda out: [x_path, mask_path, str(n), out])
        run_pair("causal_mask", lambda out: [causal_path, "4", out])
        run_pair(
            "residual_add",
            lambda out: [x_path, y_path, str(n), out],
        )
        bias = [0.1, -0.2, 0.3, 0.4]
        bias_path = os.path.join(tmp, "bias.bin")
        write_floats(bias_path, bias)
        run_pair("bias_add", lambda out: [x_path, bias_path, str(n), str(4), out])

        scale = [1.0, 0.5, 2.0, 1.5]
        scale_path = os.path.join(tmp, "scale.bin")
        write_floats(scale_path, scale)
        run_pair(
            "batch_norm",
            lambda out: [x_path, scale_path, bias_path, str(4), str(32), "1e-5", out],
        )

        h = w = 8
        pool_input = [float(i) for i in range(h * w)]
        pool_path = os.path.join(tmp, "pool.bin")
        write_floats(pool_path, pool_input)
        run_pair(
            "max_pool2d",
            lambda out: [pool_path, str(h), str(w), str(1), str(4), str(2), out],
        )
        run_pair(
            "avg_pool2d",
            lambda out: [pool_path, str(h), str(w), str(1), str(4), str(2), out],
        )

        rows, hidden = 2, 16
        ln_input = [float((i % 7) - 3) for i in range(rows * hidden)]
        ln_path = os.path.join(tmp, "ln.bin")
        write_floats(ln_path, ln_input)
        run_pair("layer_norm", lambda out: [ln_path, str(rows), str(hidden), "1e-5", out])
        run_pair(
            "rms_norm",
            lambda out: [ln_path, str(rows), str(hidden), "1e-5", out],
        )

        vocab, dim = 8, 8
        weight = [float((i * 3) % 11) for i in range(vocab * dim)]
        weight_path = os.path.join(tmp, "weight.bin")
        write_floats(weight_path, weight)
        indices = [0, 2, 5, 7]
        indices_path = os.path.join(tmp, "indices.bin")
        write_ints(indices_path, indices)
        run_pair(
            "embedding_lookup",
            lambda out: [weight_path, indices_path, str(len(indices)), str(dim), out],
        )

        rows, cols = 4, 8
        trans_input = [float(i) for i in range(rows * cols)]
        trans_path = os.path.join(tmp, "trans.bin")
        write_floats(trans_path, trans_input)
        run_pair("transpose", lambda out: [trans_path, str(rows), str(cols), out])

        m, k, n = 8, 8, 8
        a = [float((i * 5) % 13) for i in range(m * k)]
        b = [float((i * 7) % 11) for i in range(k * n)]
        a_path = os.path.join(tmp, "a.bin")
        b_path = os.path.join(tmp, "b.bin")
        write_floats(a_path, a)
        write_floats(b_path, b)
        run_pair("matmul_naive", lambda out: [a_path, b_path, str(m), str(n), str(k), out])
        run_pair("matmul_tiled", lambda out: [a_path, b_path, str(m), str(n), str(k), out])

        rows, cols = 2, 8
        sm_input = [float((i % 5) - 2) for i in range(rows * cols)]
        sm_path = os.path.join(tmp, "sm.bin")
        write_floats(sm_path, sm_input)
        run_pair("softmax_row", lambda out: [sm_path, str(rows), str(cols), out])

        red_input = [float(i + 1) for i in range(256)]
        red_path = os.path.join(tmp, "red.bin")
        write_floats(red_path, red_input)
        run_pair("reduce_sum", lambda out: [red_path, str(256), str(256), out])

        c, h, w, out_c, kh, kw, out_h, out_w = 1, 4, 4, 1, 2, 2, 3, 3
        conv_in = [float(i) for i in range(c * h * w)]
        conv_w = [0.25] * (out_c * c * kh * kw)
        conv_in_path = os.path.join(tmp, "conv_in.bin")
        conv_w_path = os.path.join(tmp, "conv_w.bin")
        write_floats(conv_in_path, conv_in)
        write_floats(conv_w_path, conv_w)
        conv_args = [
            conv_in_path,
            conv_w_path,
            str(c),
            str(h),
            str(w),
            str(out_c),
            str(kh),
            str(kw),
            str(out_h),
            str(out_w),
        ]
        run_pair("conv2d_naive", lambda out: [*conv_args, out])
        run_pair("conv2d_im2col", lambda out: [*conv_args, out])
        run_pair("conv2d_shared", lambda out: [*conv_args, out])

        batch, heads, seq, dim = 1, 2, 4, 4
        q = [float((i % 5) - 2) for i in range(batch * heads * seq * dim)]
        k = [float((i % 7) - 3) for i in range(batch * heads * seq * dim)]
        q_path = os.path.join(tmp, "q.bin")
        k_path = os.path.join(tmp, "k.bin")
        write_floats(q_path, q)
        write_floats(k_path, k)
        run_pair(
            "attention_scores",
            lambda out: [q_path, k_path, str(batch), str(heads), str(seq), str(dim), out],
        )

    print("operator kernels: PASS (supported subset)")


if __name__ == "__main__":
    main()
