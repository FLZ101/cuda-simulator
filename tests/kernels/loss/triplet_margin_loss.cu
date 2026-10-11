extern "C" __global__ void triplet_margin_loss(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  float pos = 0.0F;
  float neg = 0.0F;
  for (int i = 0; i < p1; ++i) {
    pos += (a[i] - b[i]) * (a[i] - b[i]);
    neg += (a[i] - c[i]) * (a[i] - c[i]);
  }
  out[0] = fmaxf(0.0F, pos - neg + f0);
}
