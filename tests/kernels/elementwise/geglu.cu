extern "C" __global__ void geglu(const float *a, const float *b, float *out,
                                 int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float x = a[idx];
    out[idx] = 0.5F * x * (1.0F + erfcf(-x / sqrtf(2.0F))) * b[idx];
  }
}
