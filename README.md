# cuda-simulator

A CPU-hosted simulator for small, externally generated CUDA programs.

The simulator loads PTX and executes SIMT kernels on the host CPU. The v1
targets are ResNet-18/ResNet-50 and GPT-2 124M.

See [DESIGN.md](DESIGN.md) for the full design.

## Build

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Options

```text
CUDA_SIM_ENABLE_JIT=OFF|ON
CUDA_SIM_ENABLE_SANITIZERS=OFF|ON
```
