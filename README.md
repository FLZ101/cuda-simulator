# cuda-simulator

A CPU-hosted simulator for small, externally generated CUDA programs.

The simulator loads PTX and executes SIMT kernels on the host CPU. The v1
targets are ResNet-18/ResNet-50 and GPT-2 124M.

See [DESIGN.md](DESIGN.md) for the full design.

Additional documentation:

- [CUDA_PROGRAMMING_MODEL.md](CUDA_PROGRAMMING_MODEL.md)
- [GPU_MODEL.md](GPU_MODEL.md)
- [CUDA_BEST_PRACTICES.md](CUDA_BEST_PRACTICES.md)

## Requirements

- CMake 3.24 or newer
- Ninja
- C++20 compiler
- CUDA toolkit with `nvcc`
- CUDA-capable GPU with `nvidia-smi`
- clang-format
- host C++ compiler with `-march=native` support for generated native kernels

WSL2 works like a native Linux distro for the CUDA toolkit: install the
toolkit inside the Linux filesystem; the GPU driver is supplied by the Windows
host.

## Tested environments

- Ubuntu 26.04.1 LTS (x86-64), kernel 7.0.0-38-generic, CUDA 13.3 (nvcc
  V13.3.73), driver 615.71.09, GeForce RTX 3050 Ti Laptop GPU (4096 MiB,
  `sm_86`), gcc 15.2.0, clang 21.1.8, glibc 2.43, PyTorch 2.14.1+cu130,
  Triton 3.8.0

## Build

```bash
make build
make test
```

## Options

The Makefile drives CMake with Ninja by default.

```text
make build      configure and build
make test       build and run tests
make run        build and run the CLI
make clean      remove the build directory
```

Direct CMake configuration is still available:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCUDA_SIM_ENABLE_SANITIZERS=OFF
cmake --build build
ctest --test-dir build --output-on-failure
```
