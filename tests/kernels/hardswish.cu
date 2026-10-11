extern "C" __global__ void hardswish(const float *in, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = in[i] * fminf(fmaxf(in[i] + 3.0F, 0.0F), 6.0F) / 6.0F;
  }
}
