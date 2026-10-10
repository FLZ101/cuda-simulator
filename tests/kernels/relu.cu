extern "C" __global__ void relu(const float *in, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = in[i] > 0.0F ? in[i] : 0.0F;
  }
}
