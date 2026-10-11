extern "C" __global__ void scatter_add(const float *in, const int *indices,
                                       float *out, int rows, int cols,
                                       int num_indices) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_indices * cols;
  if (idx >= total) {
    return;
  }
  int row = idx / cols;
  int col = idx % cols;
  int dst_row = indices[row];
  if (dst_row < 0 || dst_row >= rows) {
    return;
  }
  atomicAdd(&out[dst_row * cols + col], in[idx]);
}
