extern "C" __global__ void residual_add(const float *input, const float *residual,
                                        float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = input[i] + residual[i];
  }
}
