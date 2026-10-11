extern "C" __global__ void fused_mlp(const float *a, const float *b,
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
      float sum = c[j];
      for (int e = 0; e < k; ++e) {
        sum += a[i * k + e] * b[e * h + j];
      }
      const float gelu = 0.5F * sum *
                         (1.0F + tanhf(0.7978845608F *
                                       (sum + 0.044715F * sum * sum * sum)));
      for (int o = 0; o < n; ++o) {
        out[i * n + o] += gelu * d[j * n + o];
      }
    }
  }
}
