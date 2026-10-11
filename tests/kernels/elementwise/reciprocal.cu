extern "C" __global__ void reciprocal(const float *in, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = 1.0F / in[i];
  }
}
