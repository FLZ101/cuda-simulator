extern "C" __global__ void masked_fill(const float *in, const float *mask,
                                       float value, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = mask[i] != 0.0F ? value : in[i];
  }
}
