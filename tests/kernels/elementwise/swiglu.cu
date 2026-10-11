extern "C" __global__ void swiglu(const float *a, const float *b, float *out,
                                  int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = (a[i] / (1.0F + expf(-a[i]))) * b[i];
  }
}
