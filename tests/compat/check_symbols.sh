#!/bin/sh
set -eu

runtime_lib=$1
driver_lib=$2

nm -D --defined-only "$runtime_lib" | grep -q ' cudaMalloc$'
nm -D --defined-only "$runtime_lib" | grep -q ' cudaFree$'
nm -D --defined-only "$runtime_lib" | grep -q ' cudaLaunchKernel$'
nm -D --defined-only "$runtime_lib" | grep -q ' __cudaRegisterFatBinary$'

nm -D --defined-only "$driver_lib" | grep -q ' cuInit$'
nm -D --defined-only "$driver_lib" | grep -q ' cuMemAlloc$'
nm -D --defined-only "$driver_lib" | grep -q ' cuMemFree$'
nm -D --defined-only "$driver_lib" | grep -q ' cuLaunchKernel$'
