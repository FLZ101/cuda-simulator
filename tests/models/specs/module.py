#!/usr/bin/env python3

import argparse
import importlib.util
import pathlib
import re
import struct

import torch
import torch.fx
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


@triton.jit
def conv2d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    S: tl.constexpr,
    stride: tl.constexpr,
    pad: tl.constexpr,
):
    pid = tl.program_id(0)
    oh = tl.program_id(1)
    ow = tl.program_id(2)
    n = pid // K
    k = pid % K
    acc = tl.load(b_ptr + k)
    out_h = (H + 2 * pad - R) // stride + 1
    out_w = (W + 2 * pad - S) // stride + 1
    for c in tl.range(0, C):
        for r in tl.range(0, R):
            for s in tl.range(0, S):
                ih = oh * stride - pad + r
                iw = ow * stride - pad + s
                in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                x = tl.load(
                    x_ptr + ((n * C + c) * H + ih) * W + iw,
                    mask=in_bounds,
                    other=0.0,
                )
                w = tl.load(w_ptr + ((k * C + c) * R + r) * S + s)
                acc += x * w
    tl.store(
        out_ptr + ((n * K + k) * out_h + oh) * out_w + ow,
        acc,
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--module-file", required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--in-features", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--out-features", type=int, default=8)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--out-channels", type=int, default=4)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--block", type=int, default=32)


def _load_model_module(module_file: str):
    module_path = pathlib.Path(module_file).resolve()
    module_spec = importlib.util.spec_from_file_location("cudasim_user_model", module_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"cannot load model module: {module_file}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _build_model(args: argparse.Namespace):
    model_module = _load_model_module(args.module_file)
    model = model_module.build_model(args)
    model.eval()
    return model, model_module.example_input(args)


def _trace(model: torch.nn.Module):
    traced = torch.fx.symbolic_trace(model)
    traced.recompile()
    return traced


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


def _linear_nodes(traced):
    for node in traced.graph.nodes:
        if node.op != "call_module":
            continue
        module = traced.get_submodule(node.target)
        if isinstance(module, torch.nn.Linear):
            yield node, module


def _relu_nodes(traced):
    for node in traced.graph.nodes:
        if node.op != "call_module":
            continue
        module = traced.get_submodule(node.target)
        if isinstance(module, torch.nn.ReLU):
            yield node, module


def _weighted_nodes(traced):
    for node in traced.graph.nodes:
        if node.op != "call_module":
            continue
        module = traced.get_submodule(node.target)
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            yield node, module


def _predecessor_name(node):
    value = node.args[0]
    if not isinstance(value, torch.fx.Node):
        raise RuntimeError(f"unsupported input for node {node.name}")
    return value.name


def _shape_map(traced, example_input):
    shapes = {}
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            shapes[node.name] = list(example_input.shape)
        elif node.op == "output":
            continue
        elif node.op == "call_module":
            module = traced.get_submodule(node.target)
            input_shape = shapes[_predecessor_name(node)]
            if isinstance(module, torch.nn.Linear):
                if len(input_shape) != 2:
                    raise RuntimeError(f"unsupported Linear input shape: {input_shape}")
                shapes[node.name] = [input_shape[0], module.out_features]
            elif isinstance(module, torch.nn.Conv2d):
                if len(input_shape) != 4:
                    raise RuntimeError(f"unsupported Conv2d input shape: {input_shape}")
                kernel_h, kernel_w = module.kernel_size
                stride_h, stride_w = module.stride
                pad_h, pad_w = module.padding
                dilation_h, dilation_w = module.dilation
                out_h = (
                    input_shape[2] + 2 * pad_h - dilation_h * (kernel_h - 1) - 1
                ) // stride_h + 1
                out_w = (
                    input_shape[3] + 2 * pad_w - dilation_w * (kernel_w - 1) - 1
                ) // stride_w + 1
                shapes[node.name] = [
                    input_shape[0],
                    module.out_channels,
                    out_h,
                    out_w,
                ]
            elif isinstance(module, torch.nn.ReLU):
                shapes[node.name] = input_shape
            else:
                raise RuntimeError(f"unsupported module in traced graph: {type(module).__name__}")
        else:
            raise RuntimeError(f"unsupported traced node: {node.op} {node.name}")
    return shapes


def _graph_program(traced, shapes):
    program = []
    for node in traced.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        module = traced.get_submodule(node.target)
        input_name = _predecessor_name(node)
        if isinstance(module, torch.nn.Linear):
            program.append(
                {
                    "kind": "linear",
                    "name": node.name,
                    "input": input_name,
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "shape": shapes[node.name],
                }
            )
        elif isinstance(module, torch.nn.Conv2d):
            kernel_h, kernel_w = module.kernel_size
            stride_h, stride_w = module.stride
            pad_h, pad_w = module.padding
            program.append(
                {
                    "kind": "conv2d",
                    "name": node.name,
                    "input": input_name,
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "kernel_h": kernel_h,
                    "kernel_w": kernel_w,
                    "stride_h": stride_h,
                    "stride_w": stride_w,
                    "pad_h": pad_h,
                    "pad_w": pad_w,
                    "input_shape": shapes[_predecessor_name(node)],
                    "shape": shapes[node.name],
                }
            )
        elif isinstance(module, torch.nn.ReLU):
            program.append(
                {
                    "kind": "relu",
                    "name": node.name,
                    "input": input_name,
                    "shape": shapes[node.name],
                }
            )
        else:
            raise RuntimeError(f"unsupported module in traced graph: {type(module).__name__}")
    return program


def run_cpu(args: argparse.Namespace, output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1234)
    model, example_input = _build_model(args)
    with torch.no_grad():
        output = model(example_input)

    torch.save(
        {
            "x": example_input,
            "output": output,
            "shape": tuple(output.shape),
            "dtype": str(output.dtype),
        },
        output_dir / "reference.pt",
    )
    _write_floats(output_dir / "input.bin", example_input)
    _write_floats(output_dir / "cpu_output.bin", output)

    traced = _trace(model)
    for index, (_, module) in enumerate(_weighted_nodes(traced)):
        _write_floats(output_dir / f"weight{index}.bin", module.weight)
        if module.bias is not None:
            _write_floats(output_dir / f"bias{index}.bin", module.bias)


def native_arguments(args: argparse.Namespace, output_dir: pathlib.Path) -> list[str]:
    model, _ = _build_model(args)
    traced = _trace(model)
    weighted = list(_weighted_nodes(traced))
    arguments = [
        "--input",
        str(output_dir / "input.bin"),
        "--output",
        str(output_dir / "gpu_output.bin"),
    ]
    for index, _ in enumerate(weighted):
        arguments.extend(
            [
                f"--weight{index}",
                str(output_dir / f"weight{index}.bin"),
                f"--bias{index}",
                str(output_dir / f"bias{index}.bin"),
            ]
        )
    return arguments


def _buffer_name(node_name):
    return "d_" + node_name.replace(".", "_")


def _numel(shape):
    count = 1
    for dim in shape:
        count *= dim
    return count


def _linear_index(linears, op):
    return next(index for index, candidate in enumerate(linears) if candidate is op)


def _conv_index(convs, op):
    return next(index for index, candidate in enumerate(convs) if candidate is op)


def _relu_index(relus, op):
    return next(index for index, candidate in enumerate(relus) if candidate is op)


def _weighted_index(weighted_ops, op):
    return next(index for index, candidate in enumerate(weighted_ops) if candidate is op)


def generate_native_runner(args: argparse.Namespace, output_path: pathlib.Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-capable PyTorch is required to generate Triton PTX")

    model, example_input = _build_model(args)
    traced = _trace(model)
    shapes = _shape_map(traced, example_input)
    program = _graph_program(traced, shapes)
    linears = [op for op in program if op["kind"] == "linear"]
    convs = [op for op in program if op["kind"] == "conv2d"]
    relus = [op for op in program if op["kind"] == "relu"]
    weighted = [op for op in program if op["kind"] in ("linear", "conv2d")]

    ptx_by_key = {}
    for index, op in enumerate(linears):
        m = op["shape"][0]
        k = op["in_features"]
        n = op["out_features"]
        x = torch.randn(m, k, device="cuda", dtype=torch.float32)
        w = torch.randn(n, k, device="cuda", dtype=torch.float32)
        b = torch.randn(n, device="cuda", dtype=torch.float32)
        out = torch.empty(m, n, device="cuda", dtype=torch.float32)
        ptx = _compile_kernel(
            linear_kernel,
            (m, n),
            (x, w, b, out, m, n),
            {"k": k, "num_warps": 1},
        ).replace("linear_kernel", f"linear_{index}_kernel")
        ptx_by_key[f"linear_{index}"] = ptx

    for index, op in enumerate(convs):
        n, c, h, w = op["input_shape"]
        k = op["out_channels"]
        r = op["kernel_h"]
        s = op["kernel_w"]
        stride = op["stride_h"]
        pad = op["pad_h"]
        out_h = op["shape"][2]
        out_w = op["shape"][3]
        x = torch.randn(n, c, h, w, device="cuda", dtype=torch.float32)
        weight = torch.randn(k, c, r, s, device="cuda", dtype=torch.float32)
        bias = torch.randn(k, device="cuda", dtype=torch.float32)
        out = torch.empty(n, k, out_h, out_w, device="cuda", dtype=torch.float32)
        ptx = _compile_kernel(
            conv2d_kernel,
            (n * k, out_h, out_w),
            (x, weight, bias, out),
            {
                "C": c,
                "H": h,
                "W": w,
                "K": k,
                "R": r,
                "S": s,
                "stride": stride,
                "pad": pad,
                "num_warps": 1,
            },
        ).replace("conv2d_kernel", f"conv2d_{index}_kernel")
        ptx_by_key[f"conv2d_{index}"] = ptx

    for index, op in enumerate(relus):
        count = _numel(op["shape"])
        input_tensor = torch.empty(count, device="cuda", dtype=torch.float32)
        output_tensor = torch.empty(count, device="cuda", dtype=torch.float32)
        ptx = _compile_kernel(
            relu_kernel,
            (1,),
            (input_tensor, output_tensor, count),
            {"BLOCK": args.block, "num_warps": 1},
        ).replace("relu_kernel", f"relu_{index}_kernel")
        ptx_by_key[f"relu_{index}"] = ptx

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _generate_cpp(
            args,
            traced,
            program,
            linears,
            convs,
            relus,
            weighted,
            shapes,
            ptx_by_key,
        )
    )


def _generate_cpp(args, traced, program, linears, convs, relus, weighted, shapes, ptx_by_key):
    placeholder = next(node for node in traced.graph.nodes if node.op == "placeholder")
    output_node = next(node for node in traced.graph.nodes if node.op == "output")
    final_node = output_node.args[0]
    if not isinstance(final_node, torch.fx.Node):
        raise RuntimeError("traced model must return a tensor")

    function_loads = []
    for index, _ in enumerate(linears):
        function_loads.append(f"""  CUfunction linear{index}_fn = load_function(
      R"CUPTX({ptx_by_key[f"linear_{index}"]})CUPTX", "linear_{index}_kernel");""")
    for index, _ in enumerate(convs):
        function_loads.append(f"""  CUfunction conv{index}_fn = load_function(
      R"CUPTX({ptx_by_key[f"conv2d_{index}"]})CUPTX", "conv2d_{index}_kernel");""")
    for index, _ in enumerate(relus):
        function_loads.append(f"""  CUfunction relu{index}_fn = load_function(
      R"CUPTX({ptx_by_key[f"relu_{index}"]})CUPTX", "relu_{index}_kernel");""")
    declarations = "\n".join(function_loads)
    check_functions = (
        " || ".join(
            [f"!linear{index}_fn" for index in range(len(linears))]
            + [f"!conv{index}_fn" for index in range(len(convs))]
            + [f"!relu{index}_fn" for index in range(len(relus))]
        )
        or "false"
    )

    path_decls = "\n".join(
        f"  std::string weight{index}_path;\n  std::string bias{index}_path;"
        for index in range(len(weighted))
    )

    weight_arg_branches = []
    for index, _ in enumerate(weighted):
        weight_arg_branches.append(f"""    }} else if (arg == "--weight{index}" && i + 1 < argc) {{
      weight{index}_path = argv[++i];
    }} else if (arg == "--bias{index}" && i + 1 < argc) {{
      bias{index}_path = argv[++i];""")
    arg_parser = "\n".join(weight_arg_branches)

    weight_reads = []
    for index, op in enumerate(weighted):
        if op["kind"] == "linear":
            weight_count = op["out_features"] * op["in_features"]
            bias_count = op["out_features"]
        else:
            weight_count = op["out_channels"] * op["in_channels"] * op["kernel_h"] * op["kernel_w"]
            bias_count = op["out_channels"]
        weight_reads.append(f"""  const auto weight{index} = read_floats(weight{index}_path);
  const auto bias{index} = read_floats(bias{index}_path);
  if (static_cast<int>(weight{index}.size()) != {weight_count} ||
      static_cast<int>(bias{index}.size()) != {bias_count}) {{
    return 3;
  }}""")

    device_weight_allocs = "\n".join(
        f"""  CUdeviceptr d_weight{index} =
      allocate_and_copy(weight{index}.data(),
                        weight{index}.size() * sizeof(float));
  CUdeviceptr d_bias{index} =
      allocate_and_copy(bias{index}.data(), bias{index}.size() * sizeof(float));"""
        for index in range(len(weighted))
    )
    alloc_guard_parts = ["!d_input"]
    alloc_guard_parts.extend(f"!d_weight{index}" for index in range(len(weighted)))
    alloc_guard_parts.extend(f"!d_bias{index}" for index in range(len(weighted)))
    alloc_guard = " ||\n      ".join(alloc_guard_parts)

    buffer_allocs = []
    for op in program:
        var = _buffer_name(op["name"])
        buffer_allocs.append(f"""  CUdeviceptr {var} = 0;
  if (!check(cuMemAlloc(&{var}, {_numel(op["shape"])} * sizeof(float)))) {{
    return 5;
  }}""")

    launches = []
    for op in program:
        input_var = "d_input" if op["input"] == placeholder.name else _buffer_name(op["input"])
        output_var = _buffer_name(op["name"])
        if op["kind"] == "linear":
            linear_index = _linear_index(linears, op)
            weight_index = _weighted_index(weighted, op)
            launches.append(f"""  if (!launch_linear(linear{linear_index}_fn, {input_var},
                     d_weight{weight_index}, d_bias{weight_index}, {output_var},
                     {op["shape"][0]}, {op["out_features"]})) {{
    return 6;
  }}""")
        elif op["kind"] == "conv2d":
            conv_index = _conv_index(convs, op)
            weight_index = _weighted_index(weighted, op)
            n, c, h, w = op["input_shape"]
            k = op["out_channels"]
            r = op["kernel_h"]
            s = op["kernel_w"]
            stride = op["stride_h"]
            pad = op["pad_h"]
            out_h = op["shape"][2]
            out_w = op["shape"][3]
            launches.append(f"""  if (!launch_conv(conv{conv_index}_fn, {input_var},
                   d_weight{weight_index}, d_bias{weight_index}, {output_var},
                   {n}, {k}, {out_h}, {out_w})) {{
    return 8;
  }}""")
        else:
            index = _relu_index(relus, op)
            count = _numel(op["shape"])
            launches.append(f"""  CUdeviceptr relu_scratch{index} = 0;
  int relu_count{index} = {count};
  void *relu_params[] = {{&{input_var}, &{output_var}, &relu_count{index}, &relu_scratch{index}}};
  const unsigned int relu_grid =
      (static_cast<unsigned int>(relu_count{index}) + {args.block} - 1) /
      {args.block};
  if (!check(cuLaunchKernel(relu{index}_fn, relu_grid, 1, 1,
                            {args.block}, 1, 1, 0, nullptr, relu_params,
                            nullptr))) {{
    return 7;
  }}""")

    final_var = _buffer_name(final_node.name)
    final_count = _numel(shapes[final_node.name])
    placeholder_count = _numel(shapes[placeholder.name])

    launch_helpers = []
    if linears:
        launch_helpers.append(
            """bool launch_linear(CUfunction function, CUdeviceptr x, CUdeviceptr w,
                   CUdeviceptr b, CUdeviceptr out, int m, int n) {
  CUdeviceptr scratch = 0;
  void *params[] = {&x, &w, &b, &out, &m, &n, &scratch};
  return check(cuLaunchKernel(function, static_cast<unsigned int>(m),
                              static_cast<unsigned int>(n), 1, 32, 1, 1, 0,
                              nullptr, params, nullptr));
}"""
        )
    if convs:
        launch_helpers.append("""bool launch_conv(CUfunction function, CUdeviceptr x, CUdeviceptr w,
                 CUdeviceptr b, CUdeviceptr out, int n, int k, int out_h,
                 int out_w) {
  CUdeviceptr scratch = 0;
  void *params[] = {&x, &w, &b, &out, &scratch};
  return check(cuLaunchKernel(function, static_cast<unsigned int>(n * k),
                              static_cast<unsigned int>(out_h),
                              static_cast<unsigned int>(out_w), 32, 1, 1, 0,
                              nullptr, params, nullptr));
}""")
    launch_helpers_text = "\n".join(launch_helpers)

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

{launch_helpers_text}

}} // namespace

int main(int argc, char **argv) {{
  std::string input_path;
  std::string output_path;
{path_decls}

  for (int i = 1; i < argc; ++i) {{
    const std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {{
      input_path = argv[++i];
    }} else if (arg == "--output" && i + 1 < argc) {{
      output_path = argv[++i];
{arg_parser}
    }}
  }}

  if (input_path.empty() || output_path.empty()) {{
    return 1;
  }}
  if (!initialize_cuda()) {{
    return 2;
  }}

  const auto input = read_floats(input_path);
  if (static_cast<int>(input.size()) != {placeholder_count}) {{
    return 3;
  }}

{chr(10).join(weight_reads)}

{declarations}
  if ({check_functions}) {{
    return 4;
  }}

  CUdeviceptr d_input = allocate_and_copy(input.data(), input.size() * sizeof(float));
{device_weight_allocs}
  if ({alloc_guard}) {{
    return 5;
  }}

{chr(10).join(buffer_allocs)}

{chr(10).join(launches)}

  std::vector<float> result({final_count});
  if (!check(cuMemcpyDtoH(result.data(), {final_var},
                          result.size() * sizeof(float)))) {{
    return 9;
  }}
  write_floats(output_path, result);
  return 0;
}}
"""
    return source
