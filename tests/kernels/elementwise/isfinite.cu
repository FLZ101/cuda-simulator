extern "C" __global__ void isfinite_kernel(const float *in, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = isfinite(in[idx]) ? 1.0F : 0.0F;
  }
}
