extern "C" __global__ void ciou(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  constexpr float pi = 3.14159265358979323846F;
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
      const float union_area = area1 + area2 - inter + 1e-6F;
      const float iou = inter / union_area;

      const float cx1 = 0.5F * (lhs[0] + lhs[2]);
      const float cy1 = 0.5F * (lhs[1] + lhs[3]);
      const float w1 = fmaxf(0.0F, lhs[2] - lhs[0]);
      const float h1 = fmaxf(0.0F, lhs[3] - lhs[1]);
      const float cx2 = 0.5F * (rhs[0] + rhs[2]);
      const float cy2 = 0.5F * (rhs[1] + rhs[3]);
      const float w2 = fmaxf(0.0F, rhs[2] - rhs[0]);
      const float h2 = fmaxf(0.0F, rhs[3] - rhs[1]);
      const float rho2 = (cx1 - cx2) * (cx1 - cx2) +
                         (cy1 - cy2) * (cy1 - cy2);
      const float enclose_w =
          fmaxf(lhs[2], rhs[2]) - fminf(lhs[0], rhs[0]);
      const float enclose_h =
          fmaxf(lhs[3], rhs[3]) - fminf(lhs[1], rhs[1]);
      const float c2 =
          enclose_w * enclose_w + enclose_h * enclose_h + 1e-6F;
      const float atan1 = atanf(w1 / (h1 + 1e-6F));
      const float atan2 = atanf(w2 / (h2 + 1e-6F));
      const float v =
          (4.0F / (pi * pi)) * (atan2 - atan1) * (atan2 - atan1);
      const float alpha = v / ((1.0F - iou) + v + 1e-6F);
      out[i * p2 + j] = iou - rho2 / c2 - alpha * v;
    }
  }
}
