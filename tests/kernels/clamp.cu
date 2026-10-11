extern "C" __global__ void clamp(const float *in, float *out, float min_value,
                                 float max_value, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  float value = in[i];
  value = value < min_value ? min_value : value;
  value = value > max_value ? max_value : value;
  out[i] = value;
}
