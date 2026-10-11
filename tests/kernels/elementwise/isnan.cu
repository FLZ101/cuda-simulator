extern "C" __global__ void isnan_kernel(const float *in, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = isnan(in[idx]) ? 1.0F : 0.0F;
  }
}
