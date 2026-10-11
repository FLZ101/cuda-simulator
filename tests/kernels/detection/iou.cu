extern "C" __global__ void iou(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (int i = 0; i < p1; ++i) {
    const float *lhs = a + i * p3;
    for (int j = 0; j < p2; ++j) {
      const float *rhs = b + j * p3;
      const float xx1 = fmaxf(lhs[0], rhs[0]);
      const float yy1 = fmaxf(lhs[1], rhs[1]);
      const float xx2 = fminf(lhs[2], rhs[2]);
      const float yy2 = fminf(lhs[3], rhs[3]);
      const float inter =
          fmaxf(0.0F, xx2 - xx1) * fmaxf(0.0F, yy2 - yy1);
      const float area1 =
          fmaxf(0.0F, lhs[2] - lhs[0]) * fmaxf(0.0F, lhs[3] - lhs[1]);
      const float area2 =
          fmaxf(0.0F, rhs[2] - rhs[0]) * fmaxf(0.0F, rhs[3] - rhs[1]);
      out[i * p2 + j] = inter / (area1 + area2 - inter + 1e-6F);
    }
  }
}
