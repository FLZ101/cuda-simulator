extern "C" __global__ void atan2_kernel(const float *a, const float *b,
                                        float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = atan2f(a[i], b[i]);
  }
}
