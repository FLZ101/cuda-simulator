"""Shared helpers for model-runner specs."""

from dataclasses import dataclass, field
import pathlib
import re
import struct

import torch

BANNED_PATTERN = re.compile(r"\b(mma|wmma|ldmatrix|stmatrix|wgmma|tcgen05)\b")


def strip_debug_sections(ptx: str) -> str:
    for marker in ("\n\t.file", "\n\t.section"):
        if marker in ptx:
            ptx = ptx.partition(marker)[0]
    return ptx.rstrip() + "\n"


def write_floats(path: pathlib.Path, values: torch.Tensor) -> None:
    flat = values.reshape(-1).tolist()
    path.write_bytes(b"".join(struct.pack("<f", value) for value in flat))


def compile_kernel(kernel, grid, args, kwargs) -> str:
    compiled = kernel[grid](*args, **kwargs)
    ptx = strip_debug_sections(compiled.asm["ptx"])
    matches = sorted(set(BANNED_PATTERN.findall(ptx)))
    if matches:
        raise RuntimeError(f"generated PTX contains tensor-core instructions: {matches}")
    return ptx


def runner_prelude() -> str:
    return """#include <cuda.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool check(CUresult result) { return result == CUDA_SUCCESS; }

bool initialize_cuda() {
  if (!check(cuInit(0))) {
    return false;
  }
  CUdevice device = 0;
  if (!check(cuDeviceGet(&device, 0))) {
    return false;
  }
  CUcontext context = nullptr;
  if (!check(cuDevicePrimaryCtxRetain(&context, device))) {
    return false;
  }
  return check(cuCtxSetCurrent(context));
}

std::vector<float> read_floats(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  std::vector<float> values;
  float value = 0.0F;
  while (input.read(reinterpret_cast<char *>(&value), sizeof(value))) {
    values.push_back(value);
  }
  return values;
}

void write_floats(const std::string &path, const std::vector<float> &values) {
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
}

CUdeviceptr allocate_and_copy(const void *data, std::size_t bytes) {
  CUdeviceptr ptr = 0;
  if (!check(cuMemAlloc(&ptr, bytes))) {
    return 0;
  }
  if (bytes != 0 && !check(cuMemcpyHtoD(ptr, data, bytes))) {
    cuMemFree(ptr);
    return 0;
  }
  return ptr;
}

CUfunction load_function(const char *ptx, const char *name) {
  CUmodule module = nullptr;
  if (!check(cuModuleLoadData(&module, ptx))) {
    return nullptr;
  }
  CUfunction function = nullptr;
  if (!check(cuModuleGetFunction(&function, module, name))) {
    return nullptr;
  }
  return function;
}

"""


@dataclass
class Arg:
    flag: str
    var: str
    type: str = "int"


@dataclass
class Local:
    type: str
    var: str
    expr: str


@dataclass
class Input:
    var: str
    arg: str
    count: str


@dataclass
class Kernel:
    name: str
    ptx: str
    grid: tuple[str, str, str]
    block: str
    params: list[str]


@dataclass
class Runner:
    args: list[Arg] = field(default_factory=list)
    locals: list[Local] = field(default_factory=list)
    inputs: list[Input] = field(default_factory=list)
    intermediates: list[tuple[str, str]] = field(default_factory=list)
    output: str = "d_out"
    output_count: str = ""
    kernels: list[Kernel] = field(default_factory=list)


def generate_native_runner(runner: Runner, output_path: pathlib.Path) -> None:
    parts: list[str] = [runner_prelude(), "} // namespace\n\nint main(int argc, char **argv) {\n"]

    # Argument and local declarations.
    for arg in runner.args:
        if arg.type == "string":
            parts.append(f"  std::string {arg.var};\n")
        elif arg.type == "float":
            parts.append(f"  float {arg.var} = 0.0F;\n")
        else:
            parts.append(f"  int {arg.var} = 0;\n")
    parts.append("  std::string output;\n")
    parts.append("\n")

    # Command-line parsing.
    parts.append("  for (int i = 1; i < argc; ++i) {\n")
    parts.append("    const std::string arg = argv[i];\n")
    branches = []
    for arg in runner.args:
        branches.append(
            f'    if (arg == "{arg.flag}" && i + 1 < argc) {{\n'
            f"      {arg.var} = "
            + (
                "argv[++i];"
                if arg.type == "string"
                else f"std::stoi(argv[++i]);" if arg.type == "int" else f"std::stof(argv[++i]);"
            )
            + "\n    }"
        )
    branches.append(
        '    if (arg == "--output" && i + 1 < argc) {\n' "      output = argv[++i];\n" "    }"
    )
    parts.append(" else ".join(branches))
    parts.append("\n  }\n\n")

    # Argument validation.
    checks = [f"{arg.var}.empty()" for arg in runner.args if arg.type == "string"]
    checks.append("output.empty()")
    checks.extend(f"{arg.var} <= 0" for arg in runner.args if arg.type == "int")
    parts.append("  if (" + " || ".join(checks) + ") {\n")
    parts.append("    return 1;\n  }\n")
    parts.append("  if (!initialize_cuda()) {\n    return 2;\n  }\n\n")

    # Computed locals.
    for local in runner.locals:
        parts.append(f"  {local.type} {local.var} = {local.expr};\n")
    parts.append("\n")

    # Read inputs.
    for inp in runner.inputs:
        parts.append(f"  const auto {inp.var} = read_floats({inp.arg});\n")
    parts.append("\n")
    size_checks = []
    for inp in runner.inputs:
        size_checks.append(f"static_cast<int>({inp.var}.size()) != {inp.count}")
    if size_checks:
        parts.append("  if (" + " || ".join(size_checks) + ") {\n    return 3;\n  }\n\n")

    # Load kernels.
    for index, kernel in enumerate(runner.kernels):
        parts.append(
            f'  CUfunction fn{index} = load_function(R"CUPTX({kernel.ptx})CUPTX", "{kernel.name}");\n'
        )
    parts.append(
        "  if ("
        + " || ".join(f"!fn{i}" for i in range(len(runner.kernels)))
        + ") {\n    return 4;\n  }\n\n"
    )

    # Allocate inputs.
    for inp in runner.inputs:
        parts.append(
            f"  CUdeviceptr d_{inp.var} = allocate_and_copy({inp.var}.data(), {inp.var}.size() * sizeof(float));\n"
            f"  if (!d_{inp.var}) {{\n    return 5;\n  }}\n"
        )
    # Allocate intermediates.
    for var, count in runner.intermediates:
        parts.append(
            f"  CUdeviceptr {var} = 0;\n"
            f"  if (!check(cuMemAlloc(&{var}, static_cast<std::size_t>({count}) * sizeof(float)))) {{\n"
            "    return 5;\n  }\n"
        )
    # Allocate output.
    parts.append(
        f"  CUdeviceptr {runner.output} = 0;\n"
        f"  if (!check(cuMemAlloc(&{runner.output}, static_cast<std::size_t>({runner.output_count}) * sizeof(float)))) {{\n"
        "    return 5;\n  }\n\n"
    )

    # Launch kernels.
    for index, kernel in enumerate(runner.kernels):
        params = ", ".join(kernel.params + ["&scratch", "&profile_scratch"])
        parts.append("  {\n")
        parts.append("    CUdeviceptr scratch = 0;\n")
        parts.append("    CUdeviceptr profile_scratch = 0;\n")
        parts.append(f"    void *params[] = {{{params}}};\n")
        parts.append(
            f"    if (!check(cuLaunchKernel(fn{index}, {kernel.grid[0]}, {kernel.grid[1]}, {kernel.grid[2]}, "
            f"{kernel.block}, 1, 1, 0, nullptr, params, nullptr))) {{\n"
            "      return 6;\n    }\n  }\n"
        )
    parts.append("\n")

    # Copy output back and write it.
    parts.append(
        f"  std::vector<float> result(static_cast<std::size_t>({runner.output_count}));\n"
        f"  if (!check(cuMemcpyDtoH(result.data(), {runner.output}, result.size() * sizeof(float)))) {{\n"
        "    return 7;\n  }\n"
        "  write_floats(output, result);\n"
    )

    # Free device allocations.
    for inp in runner.inputs:
        parts.append(f"  cuMemFree(d_{inp.var});\n")
    for var, _ in runner.intermediates:
        parts.append(f"  cuMemFree({var});\n")
    parts.append(f"  cuMemFree({runner.output});\n")
    parts.append("  return 0;\n}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(parts))
