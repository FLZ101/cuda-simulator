extern "C" __global__ void box_encode(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (int i = 0; i < p1; ++i) {
    const float px = 0.5F * (a[i * p2 + 0] + a[i * p2 + 2]);
    const float py = 0.5F * (a[i * p2 + 1] + a[i * p2 + 3]);
    const float pw = fmaxf(a[i * p2 + 2] - a[i * p2 + 0], 1e-6F);
    const float ph = fmaxf(a[i * p2 + 3] - a[i * p2 + 1], 1e-6F);
    const float gx = 0.5F * (b[i * p2 + 0] + b[i * p2 + 2]);
    const float gy = 0.5F * (b[i * p2 + 1] + b[i * p2 + 3]);
    const float gw = fmaxf(b[i * p2 + 2] - b[i * p2 + 0], 1e-6F);
    const float gh = fmaxf(b[i * p2 + 3] - b[i * p2 + 1], 1e-6F);
    out[i * 4 + 0] = (gx - px) / pw;
    out[i * 4 + 1] = (gy - py) / ph;
    out[i * 4 + 2] = __logf(gw / pw);
    out[i * 4 + 3] = __logf(gh / ph);
  }
}
