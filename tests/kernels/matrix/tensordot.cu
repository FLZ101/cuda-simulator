extern "C" __global__ void tensordot(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int d0 = p1;
  const int d1 = p2;
  const int contract = p3;
  const int d2 = p4;
  for (int i = 0; i < d0; ++i) {
    for (int j = 0; j < d1; ++j) {
      for (int l = 0; l < d2; ++l) {
        float value = 0.0F;
        for (int k = 0; k < contract; ++k) {
          value += a[(i * d1 + j) * contract + k] * b[k * d2 + l];
        }
        out[(i * d1 + j) * d2 + l] = value;
      }
    }
  }
}
