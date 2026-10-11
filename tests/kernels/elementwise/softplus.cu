extern "C" __global__ void softplus(const float *in, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = logf(1.0F + expf(in[i]));
  }
}
