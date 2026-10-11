extern "C" __global__ void relu6(const float *in, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = fminf(fmaxf(in[i], 0.0F), 6.0F);
  }
}
