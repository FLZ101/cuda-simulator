extern "C" __global__ void sequence_mask(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int batch = p1;
  const int max_len = p2;
  for (int i = 0; i < batch; ++i) {
    const int length = idx == nullptr ? max_len : idx[i];
    for (int j = 0; j < max_len; ++j) {
      out[i * max_len + j] = j < length ? 1.0F : 0.0F;
    }
  }
}
