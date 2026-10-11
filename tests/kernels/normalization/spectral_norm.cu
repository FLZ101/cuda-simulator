extern "C" __global__ void spectral_norm(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int rows = p1;
  const int cols = p2;
  for (int i = 0; i < rows; ++i) {
    float u = 0.0F;
    for (int j = 0; j < cols; ++j) {
      u += a[i * cols + j] * d[j];
    }
    out[i] = u;
  }
  float u_norm = 0.0F;
  for (int i = 0; i < rows; ++i) {
    u_norm += out[i] * out[i];
  }
  u_norm = sqrtf(fmaxf(u_norm, 1e-12F));
  for (int i = 0; i < rows; ++i) {
    out[i] /= u_norm;
  }
  for (int j = 0; j < cols; ++j) {
    float v = 0.0F;
    for (int i = 0; i < rows; ++i) {
      v += a[i * cols + j] * out[i];
    }
    out[rows + j] = v;
  }
  float sigma = 0.0F;
  for (int j = 0; j < cols; ++j) {
    sigma += out[rows + j] * out[rows + j];
  }
  sigma = sqrtf(fmaxf(sigma, 1e-12F));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      out[i * cols + j] = a[i * cols + j] / sigma;
    }
  }
}
