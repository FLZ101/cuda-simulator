extern "C" __global__ void box_decode(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (int i = 0; i < p1; ++i) {
    const float ax = a[i * p2 + 0];
    const float ay = a[i * p2 + 1];
    const float aw = a[i * p2 + 2];
    const float ah = a[i * p2 + 3];
    const float dx = b[i * p2 + 0];
    const float dy = b[i * p2 + 1];
    const float dw = b[i * p2 + 2];
    const float dh = b[i * p2 + 3];
    const float cx = ax + aw * dx;
    const float cy = ay + ah * dy;
    const float w = fmaxf(aw * __expf(dw), 0.0F);
    const float h = fmaxf(ah * __expf(dh), 0.0F);
    out[i * 4 + 0] = cx - 0.5F * w;
    out[i * 4 + 1] = cy - 0.5F * h;
    out[i * 4 + 2] = cx + 0.5F * w;
    out[i * 4 + 3] = cy + 0.5F * h;
  }
}
