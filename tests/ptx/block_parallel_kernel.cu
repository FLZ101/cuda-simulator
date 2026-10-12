extern "C" __global__ void block_parallel_kernel(const float *in, float *out,
                                                 int n, int iterations) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  float acc = in[i];
  for (int k = 0; k < iterations; ++k) {
    acc = acc * 0.999999f + 0.000001f;
  }
  out[i] = acc;
}
