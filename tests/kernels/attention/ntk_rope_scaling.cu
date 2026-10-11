extern "C" __global__ void ntk_rope_scaling(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (int i = 0; i < p1; ++i) {
    const float x = a[2 * i];
    const float y = a[2 * i + 1];
    const float angle = f0 * atan2f(y, x);
    const float cs = __cosf(angle);
    const float sn = __sinf(angle);
    out[2 * i] = x * cs - y * sn;
    out[2 * i + 1] = x * sn + y * cs;
  }
}
