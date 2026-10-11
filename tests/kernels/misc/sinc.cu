extern "C" __global__ void sinc_kernel(const float *a, const float *b,
                                       const float *c, const float *d,
                                       const int *idx, float *out, int p0,
                                       int p1, int p2, int p3, int p4,
                                       int p5, int p6, int p7, float f0,
                                       float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (int i = 0; i < p0; ++i) {
    const float x = a[i];
    out[i] = fabsf(x) < 1e-6F
                 ? 1.0F
                 : __sinf(3.14159265358979323846F * x) /
                       (3.14159265358979323846F * x);
  }
}
