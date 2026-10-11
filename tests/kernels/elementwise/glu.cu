extern "C" __global__ void glu(const float *a, const float *b, float *out,
                               int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] * (1.0F / (1.0F + expf(-b[idx])));
  }
}
