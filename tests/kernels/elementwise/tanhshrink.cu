extern "C" __global__ void tanhshrink(const float *in, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx] - tanhf(in[idx]);
  }
}
