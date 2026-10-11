extern "C" __global__ void softshrink(const float *in, float *out, int n,
                                      float lambda) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float x = in[idx];
    out[idx] = x > lambda ? x - lambda : (x < -lambda ? x + lambda : 0.0F);
  }
}
