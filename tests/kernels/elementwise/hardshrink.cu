extern "C" __global__ void hardshrink(const float *in, float *out, int n,
                                      float lambda) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = fabsf(in[idx]) > lambda ? in[idx] : 0.0F;
  }
}
