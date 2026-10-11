extern "C" __global__ void scale(const float *in, float *out, float factor,
                                 int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = in[i] * factor;
  }
}
