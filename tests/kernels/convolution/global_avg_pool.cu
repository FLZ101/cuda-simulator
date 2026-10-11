extern "C" __global__ void global_avg_pool(const float *in, float *out,
                                           int channels, int size) {
  int c = blockIdx.x;
  if (c >= channels) {
    return;
  }
  float sum = 0.0F;
  for (int i = 0; i < size; ++i) {
    sum += in[c * size + i];
  }
  out[c] = sum / static_cast<float>(size);
}
