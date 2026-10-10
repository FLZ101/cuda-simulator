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
- clang-format
- optional LLVM development package when `CUDA_SIM_ENABLE_JIT=ON`

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
make install-hooks  configure Git pre-commit hooks
make format     format C++ sources with clang-format
make clean      remove the build directory
```

Install the Git pre-commit hook:

```bash
make install-hooks
```

The hook runs `clang-format` on staged C/C++ files using the LLVM style.

Direct CMake configuration is still available:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCUDA_SIM_ENABLE_JIT=OFF \
  -DCUDA_SIM_ENABLE_SANITIZERS=OFF
cmake --build build
ctest --test-dir build --output-on-failure
```
