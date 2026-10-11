extern "C" __global__ void digamma_kernel(const float *a, const float *b,
                                          const float *c, const float *d,
                                          const int *idx, float *out, int p0,
                                          int p1, int p2, int p3, int p4,
                                          int p5, int p6, int p7, float f0,
                                          float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (int i = 0; i < p0; ++i) {
    float z = a[i];
    float result = 0.0F;
    while (z < 10.0F) {
      result -= 1.0F / z;
      z += 1.0F;
    }
    const float inv = 1.0F / z;
    const float inv2 = inv * inv;
    result += logf(z) - 0.5F * inv - inv2 / 12.0F +
              inv2 * inv2 / 120.0F - inv2 * inv2 * inv2 / 252.0F +
              inv2 * inv2 * inv2 * inv2 / 240.0F;
    out[i] = result;
  }
}
