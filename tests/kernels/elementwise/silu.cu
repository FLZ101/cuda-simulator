extern "C" __global__ void silu(const float *in, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = in[i];
    out[i] = x / (1.0F + expf(-x));
  }
}
