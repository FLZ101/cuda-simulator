extern "C" __global__ void copysign_kernel(const float *a, const float *b,
                                           float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = copysignf(a[i], b[i]);
  }
}
