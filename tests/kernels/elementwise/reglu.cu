extern "C" __global__ void reglu(const float *a, const float *b, float *out,
                                 int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = fmaxf(a[idx], 0.0F) * b[idx];
  }
}
