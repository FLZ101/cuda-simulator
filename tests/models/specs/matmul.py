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
def matmul_scalar_kernel(a_ptr, b_ptr, out_ptr, m, n, k: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    acc = 0.0
    for kk in tl.range(0, k):
        a = tl.load(a_ptr + row * k + kk)
        b = tl.load(b_ptr + kk * n + col)
        acc += a * b
    tl.store(out_ptr + row * n + col, acc)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--k", type=int, default=8)


def _strip_debug_sections(ptx: str) -> str:
    for marker in ("\n\t.file", "\n\t.section"):
        if marker in ptx:
            ptx = ptx.partition(marker)[0]
    return ptx.rstrip() + "\n"


def _generate_ptx(m: int, n: int, k: int) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-capable PyTorch is required to generate Triton PTX")

    a = torch.randn(m, k, device="cuda", dtype=torch.float32)
    b = torch.randn(k, n, device="cuda", dtype=torch.float32)
    out = torch.empty(m, n, device="cuda", dtype=torch.float32)
    matmul_scalar_kernel[(m, n)](a, b, out, m, n, k=k, num_warps=1)

    device_cache = matmul_scalar_kernel.device_caches[0]
    compiled_mapping = device_cache[0]
    compiled = next(iter(compiled_mapping.values()))
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
    a = torch.randn(args.m, args.k, dtype=torch.float32)
    b = torch.randn(args.k, args.n, dtype=torch.float32)
    with torch.no_grad():
        output = torch.matmul(a, b)

    torch.save(
        {
            "a": a,
            "b": b,
            "output": output,
            "shape": tuple(output.shape),
            "dtype": str(output.dtype),
        },
        output_dir / "reference.pt",
    )
    _write_floats(output_dir / "a.bin", a)
    _write_floats(output_dir / "b.bin", b)
    _write_floats(output_dir / "cpu_output.bin", output)


def native_arguments(args: argparse.Namespace, output_dir: pathlib.Path) -> list[str]:
    return [
        "--input-a",
        str(output_dir / "a.bin"),
        "--input-b",
        str(output_dir / "b.bin"),
        "--output",
        str(output_dir / "gpu_output.bin"),
        "--m",
        str(args.m),
        "--n",
        str(args.n),
        "--k",
        str(args.k),
    ]


def generate_native_runner(args: argparse.Namespace, output_path: pathlib.Path) -> None:
    ptx = _generate_ptx(args.m, args.n, args.k)
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

CUfunction load_function() {{
  static const char *const ptx = R"CUPTX({ptx})CUPTX";
  CUmodule module = nullptr;
  if (!check(cuModuleLoadData(&module, ptx))) {{
    return nullptr;
  }}
  CUfunction function = nullptr;
  if (!check(cuModuleGetFunction(&function, module, "matmul_scalar_kernel"))) {{
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

}} // namespace

int main(int argc, char **argv) {{
  std::string input_a;
  std::string input_b;
  std::string output;
  int m = 0;
  int n = 0;
  int k = 0;

  for (int i = 1; i < argc; ++i) {{
    const std::string arg = argv[i];
    if (arg == "--input-a" && i + 1 < argc) {{
      input_a = argv[++i];
    }} else if (arg == "--input-b" && i + 1 < argc) {{
      input_b = argv[++i];
    }} else if (arg == "--output" && i + 1 < argc) {{
      output = argv[++i];
    }} else if (arg == "--m" && i + 1 < argc) {{
      m = std::stoi(argv[++i]);
    }} else if (arg == "--n" && i + 1 < argc) {{
      n = std::stoi(argv[++i]);
    }} else if (arg == "--k" && i + 1 < argc) {{
      k = std::stoi(argv[++i]);
    }}
  }}

  if (input_a.empty() || input_b.empty() || output.empty() || m <= 0 ||
      n <= 0 || k <= 0) {{
    return 1;
  }}
  if (!initialize_cuda()) {{
    return 2;
  }}

  const auto a = read_floats(input_a);
  const auto b = read_floats(input_b);
  if (static_cast<int>(a.size()) != m * k ||
      static_cast<int>(b.size()) != k * n) {{
    return 3;
  }}

  CUfunction function = load_function();
  if (!function) {{
    return 4;
  }}

  CUdeviceptr d_a = allocate_and_copy(a.data(), a.size() * sizeof(float));
  CUdeviceptr d_b = allocate_and_copy(b.data(), b.size() * sizeof(float));
  CUdeviceptr d_out = 0;
  if (!d_a || !d_b ||
      !check(cuMemAlloc(&d_out,
                        static_cast<std::size_t>(m) * n * sizeof(float)))) {{
    return 5;
  }}

  CUdeviceptr scratch = 0;
  CUdeviceptr profile_scratch = 0;
  void *params[] = {{&d_a, &d_b, &d_out, &m, &n, &scratch, &profile_scratch}};
  if (!check(cuLaunchKernel(function, static_cast<unsigned int>(m),
                            static_cast<unsigned int>(n), 1, 32, 1, 1, 0,
                            nullptr, params, nullptr))) {{
    return 6;
  }}

  std::vector<float> result(static_cast<std::size_t>(m * n));
  if (!check(cuMemcpyDtoH(result.data(), d_out, result.size() * sizeof(float)))) {{
    return 7;
  }}
  write_floats(output, result);

  cuMemFree(d_a);
  cuMemFree(d_b);
  cuMemFree(d_out);
  return 0;
}}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source)
