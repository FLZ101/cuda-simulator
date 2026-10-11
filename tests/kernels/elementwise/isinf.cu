extern "C" __global__ void isinf_kernel(const float *in, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = isinf(in[idx]) ? 1.0F : 0.0F;
  }
}
