extern "C" __global__ void gated_mlp(const float *a, const float *b,
                                     const float *c, const float *d,
                                     const int *idx, float *out, int p0,
                                     int p1, int p2, int p3, int p4, int p5,
                                     int p6, int p7, float f0, float f1,
                                     float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (int i = 0; i < p0; ++i) {
    out[i] = 0.0F;
  }
  const int m = p1;
  const int k = p2;
  const int h = p3;
  const int n = p4;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < h; ++j) {
      float gate = 0.0F;
      float up = 0.0F;
      for (int e = 0; e < k; ++e) {
        gate += a[i * k + e] * b[e * h + j];
        up += a[i * k + e] * c[e * h + j];
      }
      const float act = up / (1.0F + __expf(-gate));
      for (int o = 0; o < n; ++o) {
        out[i * n + o] += act * d[j * n + o];
      }
    }
  }
}
