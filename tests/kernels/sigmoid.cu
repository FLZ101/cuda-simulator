extern "C" __global__ void sigmoid(const float *in, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = 1.0F / (1.0F + expf(-in[i]));
  }
}
