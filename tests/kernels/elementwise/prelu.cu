extern "C" __global__ void prelu(const float *in, float *out, int n,
                                 float slope) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx] > 0.0F ? in[idx] : in[idx] * slope;
  }
}
