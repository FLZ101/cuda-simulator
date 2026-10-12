#!/usr/bin/env bash
#
# Compare single-threaded and multi-threaded executor modes for a kernel with
# a large grid. The benchmark binary is linked against the cuda_shim, so the
# CUDA_SIM_EXEC_THREADS environment variable controls block-level parallelism.

set -euo pipefail

bench="${1:?usage: $0 /path/to/block_parallel_bench [bench args...]}"
shift

single_out=$(CUDA_SIM_EXEC_THREADS=1 "$bench" "$@")
multi_out=$(CUDA_SIM_EXEC_THREADS=auto "$bench" "$@")

extract() {
  sed -n "s/.*$1=\([0-9.eE+-]*\).*/\1/p" <<<"$2"
}

single_ms=$(extract "avg_ms" "$single_out")
multi_ms=$(extract "avg_ms" "$multi_out")
single_sum=$(extract "checksum" "$single_out")
multi_sum=$(extract "checksum" "$multi_out")

echo "single: $single_out"
echo "multi:  $multi_out"

if [ "$single_sum" != "$multi_sum" ]; then
  echo "checksum mismatch: single=$single_sum multi=$multi_sum" >&2
  exit 1
fi

speedup=$(awk -v a="$single_ms" -v b="$multi_ms" \
  'BEGIN { printf "%.2f", a / b }')
echo "speedup: ${speedup}x"
