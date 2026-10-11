extern "C" __global__ void fused_rms_norm_linear(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (int i = 0; i < p0; ++i) {
    out[i] = 0.0F;
  }
  const int rows = p1;
  const int hidden = p2;
  const int out_dim = p3;
  for (int r = 0; r < rows; ++r) {
    float ss = 0.0F;
    for (int e = 0; e < hidden; ++e) {
      ss += a[r * hidden + e] * a[r * hidden + e];
    }
    const float rstd = rsqrtf(ss / static_cast<float>(hidden) + 1e-5F);
    for (int o = 0; o < out_dim; ++o) {
      float sum = d == nullptr ? 0.0F : d[o];
      for (int e = 0; e < hidden; ++e) {
        sum += a[r * hidden + e] * rstd * c[e] * b[e * out_dim + o];
      }
      out[r * out_dim + o] = sum;
    }
  }
}
