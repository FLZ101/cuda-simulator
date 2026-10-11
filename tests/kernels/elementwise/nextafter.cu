extern "C" __global__ void nextafter_kernel(const float *a, const float *b,
                                            float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = nextafterf(a[idx], b[idx]);
  }
}
