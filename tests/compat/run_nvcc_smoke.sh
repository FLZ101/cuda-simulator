#!/bin/sh
set -eu

binary=$1
cudart_shim=$2
cuda_shim=$3
core_dir=$4

LD_LIBRARY_PATH="$core_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
LD_PRELOAD="$cudart_shim:$cuda_shim" \
  "$binary"
