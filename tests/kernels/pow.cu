extern "C" __global__ void pow_kernel(const float *in, float *out, float exponent,
                                      int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = powf(in[i], exponent);
  }
}
