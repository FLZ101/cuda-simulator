extern "C" __global__ void sign(const float *in, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = (in[idx] > 0.0F) - (in[idx] < 0.0F);
  }
}
