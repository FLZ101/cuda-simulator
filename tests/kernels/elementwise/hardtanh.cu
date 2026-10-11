extern "C" __global__ void hardtanh(const float *in, float *out, int n,
                                    float min_value, float max_value) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = fminf(fmaxf(in[idx], min_value), max_value);
  }
}
