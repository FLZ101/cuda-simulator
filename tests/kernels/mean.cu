extern "C" __global__ void mean(const float *in, float *out, int n) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  float sum = 0.0F;
  for (int i = 0; i < n; ++i) {
    sum += in[i];
  }
  out[0] = sum / static_cast<float>(n);
}
