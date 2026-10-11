#!/usr/bin/env python3

import argparse
import pathlib
import re
import struct

import torch
import triton
import triton.language as tl

BANNED_PATTERN = re.compile(r"\b(mma|wmma|ldmatrix|stmatrix|wgmma|tcgen05)\b")


@triton.jit
def linear_kernel(x_ptr, w_ptr, b_ptr, out_ptr, m, n, k: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    acc = tl.load(b_ptr + col)
    for kk in tl.range(0, k):
        x = tl.load(x_ptr + row * k + kk)
        w = tl.load(w_ptr + col * k + kk)
        acc += x * w
    tl.store(out_ptr + row * n + col, acc)


@triton.jit
def relu_kernel(in_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, tl.maximum(x, 0.0), mask=mask)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--in-features", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--out-features", type=int, default=8)
    parser.add_argument("--block", type=int, default=32)


def _strip_debug_sections(ptx: str) -> str:
    for marker in ("\n\t.file", "\n\t.section"):
        if marker in ptx:
            ptx = ptx.partition(marker)[0]
    return ptx.rstrip() + "\n"


def _compile_kernel(kernel, grid, args, kwargs):
    compiled = kernel[grid](*args, **kwargs)
    ptx = _strip_debug_sections(compiled.asm["ptx"])
    matches = sorted(set(BANNED_PATTERN.findall(ptx)))
    if matches:
        raise RuntimeError(f"generated PTX contains tensor-core instructions: {matches}")
    return ptx


def _write_floats(path: pathlib.Path, values: torch.Tensor) -> None:
    flat = values.reshape(-1).tolist()
    path.write_bytes(b"".join(struct.pack("<f", value) for value in flat))


def run_cpu(args: argparse.Namespace, output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1234)
    model = torch.nn.Sequential(
        torch.nn.Linear(args.in_features, args.hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden, args.out_features),
    )
    model.eval()
    x = torch.randn(args.batch, args.in_features, dtype=torch.float32)
    with torch.no_grad():
        output = model(x)

    linear0 = model[0]
    linear1 = model[2]
    torch.save(
        {
            "x": x,
            "output": output,
            "shape": tuple(output.shape),
            "dtype": str(output.dtype),
        },
        output_dir / "reference.pt",
    )
    _write_floats(output_dir / "input.bin", x)
    _write_floats(output_dir / "weight0.bin", linear0.weight)
    _write_floats(output_dir / "bias0.bin", linear0.bias)
    _write_floats(output_dir / "weight1.bin", linear1.weight)
    _write_floats(output_dir / "bias1.bin", linear1.bias)
    _write_floats(output_dir / "cpu_output.bin", output)


def native_arguments(args: argparse.Namespace, output_dir: pathlib.Path) -> list[str]:
    return [
        "--input",
        str(output_dir / "input.bin"),
        "--weight0",
        str(output_dir / "weight0.bin"),
        "--bias0",
        str(output_dir / "bias0.bin"),
        "--weight1",
        str(output_dir / "weight1.bin"),
        "--bias1",
        str(output_dir / "bias1.bin"),
        "--output",
        str(output_dir / "gpu_output.bin"),
        "--batch",
        str(args.batch),
        "--in-features",
        str(args.in_features),
        "--hidden",
        str(args.hidden),
        "--out-features",
        str(args.out_features),
    ]


def generate_native_runner(args: argparse.Namespace, output_path: pathlib.Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-capable PyTorch is required to generate Triton PTX")

    batch = args.batch
    in_features = args.in_features
    hidden = args.hidden
    out_features = args.out_features

    x = torch.randn(batch, in_features, device="cuda", dtype=torch.float32)
    w0 = torch.randn(hidden, in_features, device="cuda", dtype=torch.float32)
    b0 = torch.randn(hidden, device="cuda", dtype=torch.float32)
    mid = torch.empty(batch, hidden, device="cuda", dtype=torch.float32)
    linear0_ptx = _compile_kernel(
        linear_kernel,
        (batch, hidden),
        (x, w0, b0, mid, batch, hidden),
        {"k": in_features, "num_warps": 1},
    ).replace("linear_kernel", "linear0_kernel")

    w1 = torch.randn(out_features, hidden, device="cuda", dtype=torch.float32)
    b1 = torch.randn(out_features, device="cuda", dtype=torch.float32)
    out0 = torch.empty(batch, out_features, device="cuda", dtype=torch.float32)
    linear1_ptx = _compile_kernel(
        linear_kernel,
        (batch, out_features),
        (mid, w1, b1, out0, batch, out_features),
        {"k": hidden, "num_warps": 1},
    ).replace("linear_kernel", "linear1_kernel")

    act_in = torch.empty_like(mid)
    act_out = torch.empty_like(mid)
    relu_ptx = _compile_kernel(
        relu_kernel,
        (1,),
        (act_in, act_out, mid.numel()),
        {"BLOCK": args.block, "num_warps": 1},
    )

    source = f"""#include <cuda.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {{

bool check(CUresult result) {{ return result == CUDA_SUCCESS; }}

bool initialize_cuda() {{
  if (!check(cuInit(0))) {{
    return false;
  }}
  CUdevice device = 0;
  if (!check(cuDeviceGet(&device, 0))) {{
    return false;
  }}
  CUcontext context = nullptr;
  if (!check(cuDevicePrimaryCtxRetain(&context, device))) {{
    return false;
  }}
  return check(cuCtxSetCurrent(context));
}}

std::vector<float> read_floats(const std::string &path) {{
  std::ifstream input(path, std::ios::binary);
  std::vector<float> values;
  float value = 0.0F;
  while (input.read(reinterpret_cast<char *>(&value), sizeof(value))) {{
    values.push_back(value);
  }}
  return values;
}}

void write_floats(const std::string &path, const std::vector<float> &values) {{
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
}}

CUfunction load_function(const char *ptx, const char *name) {{
  CUmodule module = nullptr;
  if (!check(cuModuleLoadData(&module, ptx))) {{
    return nullptr;
  }}
  CUfunction function = nullptr;
  if (!check(cuModuleGetFunction(&function, module, name))) {{
    return nullptr;
  }}
  return function;
}}

CUdeviceptr allocate_and_copy(const void *data, std::size_t bytes) {{
  CUdeviceptr ptr = 0;
  if (!check(cuMemAlloc(&ptr, bytes))) {{
    return 0;
  }}
  if (bytes != 0 && !check(cuMemcpyHtoD(ptr, data, bytes))) {{
    cuMemFree(ptr);
    return 0;
  }}
  return ptr;
}}

bool launch_linear(CUfunction function, CUdeviceptr x, CUdeviceptr w,
                   CUdeviceptr b, CUdeviceptr out, int m, int n) {{
  CUdeviceptr scratch = 0;
  void *params[] = {{&x, &w, &b, &out, &m, &n, &scratch}};
  return check(cuLaunchKernel(function, static_cast<unsigned int>(m),
                              static_cast<unsigned int>(n), 1, 32, 1, 1, 0,
                              nullptr, params, nullptr));
}}

}} // namespace

int main(int argc, char **argv) {{
  std::string input_path;
  std::string weight0_path;
  std::string bias0_path;
  std::string weight1_path;
  std::string bias1_path;
  std::string output_path;
  int batch = 0;
  int in_features = 0;
  int hidden = 0;
  int out_features = 0;

  for (int i = 1; i < argc; ++i) {{
    const std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {{
      input_path = argv[++i];
    }} else if (arg == "--weight0" && i + 1 < argc) {{
      weight0_path = argv[++i];
    }} else if (arg == "--bias0" && i + 1 < argc) {{
      bias0_path = argv[++i];
    }} else if (arg == "--weight1" && i + 1 < argc) {{
      weight1_path = argv[++i];
    }} else if (arg == "--bias1" && i + 1 < argc) {{
      bias1_path = argv[++i];
    }} else if (arg == "--output" && i + 1 < argc) {{
      output_path = argv[++i];
    }} else if (arg == "--batch" && i + 1 < argc) {{
      batch = std::stoi(argv[++i]);
    }} else if (arg == "--in-features" && i + 1 < argc) {{
      in_features = std::stoi(argv[++i]);
    }} else if (arg == "--hidden" && i + 1 < argc) {{
      hidden = std::stoi(argv[++i]);
    }} else if (arg == "--out-features" && i + 1 < argc) {{
      out_features = std::stoi(argv[++i]);
    }}
  }}

  if (input_path.empty() || weight0_path.empty() || bias0_path.empty() ||
      weight1_path.empty() || bias1_path.empty() || output_path.empty() ||
      batch <= 0 || in_features <= 0 || hidden <= 0 || out_features <= 0) {{
    return 1;
  }}
  if (!initialize_cuda()) {{
    return 2;
  }}

  const auto input = read_floats(input_path);
  const auto weight0 = read_floats(weight0_path);
  const auto bias0 = read_floats(bias0_path);
  const auto weight1 = read_floats(weight1_path);
  const auto bias1 = read_floats(bias1_path);
  if (static_cast<int>(input.size()) != batch * in_features ||
      static_cast<int>(weight0.size()) != hidden * in_features ||
      static_cast<int>(bias0.size()) != hidden ||
      static_cast<int>(weight1.size()) != out_features * hidden ||
      static_cast<int>(bias1.size()) != out_features) {{
    return 3;
  }}

  CUfunction linear0_fn = load_function(
      R"CUPTX({linear0_ptx})CUPTX", "linear0_kernel");
  CUfunction linear1_fn = load_function(
      R"CUPTX({linear1_ptx})CUPTX", "linear1_kernel");
  CUfunction relu_fn = load_function(
      R"CUPTX({relu_ptx})CUPTX", "relu_kernel");
  if (!linear0_fn || !linear1_fn || !relu_fn) {{
    return 4;
  }}

  CUdeviceptr d_input =
      allocate_and_copy(input.data(), input.size() * sizeof(float));
  CUdeviceptr d_weight0 =
      allocate_and_copy(weight0.data(), weight0.size() * sizeof(float));
  CUdeviceptr d_bias0 =
      allocate_and_copy(bias0.data(), bias0.size() * sizeof(float));
  CUdeviceptr d_weight1 =
      allocate_and_copy(weight1.data(), weight1.size() * sizeof(float));
  CUdeviceptr d_bias1 =
      allocate_and_copy(bias1.data(), bias1.size() * sizeof(float));
  CUdeviceptr d_mid = 0;
  CUdeviceptr d_act = 0;
  CUdeviceptr d_out = 0;
  if (!d_input || !d_weight0 || !d_bias0 || !d_weight1 || !d_bias1 ||
      !check(cuMemAlloc(&d_mid,
                        static_cast<std::size_t>(batch) * hidden *
                            sizeof(float))) ||
      !check(cuMemAlloc(&d_act,
                        static_cast<std::size_t>(batch) * hidden *
                            sizeof(float))) ||
      !check(cuMemAlloc(&d_out,
                        static_cast<std::size_t>(batch) * out_features *
                            sizeof(float)))) {{
    return 5;
  }}

  if (!launch_linear(linear0_fn, d_input, d_weight0, d_bias0, d_mid, batch,
                     hidden)) {{
    return 6;
  }}

  CUdeviceptr scratch = 0;
  int act_count = batch * hidden;
  void *relu_params[] = {{&d_mid, &d_act, &act_count, &scratch}};
  const unsigned int relu_grid =
      (static_cast<unsigned int>(act_count) + {args.block} - 1) /
      {args.block};
  if (!check(cuLaunchKernel(relu_fn, relu_grid, 1, 1, {args.block}, 1, 1, 0,
                            nullptr, relu_params, nullptr))) {{
    return 7;
  }}

  if (!launch_linear(linear1_fn, d_act, d_weight1, d_bias1, d_out, batch,
                     out_features)) {{
    return 8;
  }}

  std::vector<float> result(static_cast<std::size_t>(batch * out_features));
  if (!check(cuMemcpyDtoH(result.data(), d_out,
                          result.size() * sizeof(float)))) {{
    return 9;
  }}
  write_floats(output_path, result);
  return 0;
}}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source)
