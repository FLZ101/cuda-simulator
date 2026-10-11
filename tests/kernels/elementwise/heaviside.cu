extern "C" __global__ void heaviside(const float *in, float *out, int n,
                                     float zero_value, float one_value) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx] <= 0.0F ? zero_value : one_value;
  }
}
