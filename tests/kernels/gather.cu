extern "C" __global__ void gather(const float *in, const int *indices,
                                  float *out, int num_indices, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_indices * cols;
  if (idx >= total) {
    return;
  }
  int row = idx / cols;
  int col = idx % cols;
  int src_row = indices[row];
  out[idx] = in[src_row * cols + col];
}
