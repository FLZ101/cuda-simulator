extern "C" __global__ void remainder_kernel(const float *a, const float *b,
                                            float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = remainderf(a[idx], b[idx]);
  }
}
