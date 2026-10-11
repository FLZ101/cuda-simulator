extern "C" __global__ void rot90(const float *a, const float *b,
                                 const float *c, const float *d,
                                 const int *idx, float *out, int p0, int p1,
                                 int p2, int p3, int p4, int p5, int p6,
                                 int p7, float f0, float f1, float f2,
                                 float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (int i = 0; i < p1; ++i) {
    for (int j = 0; j < p2; ++j) {
      out[j * p1 + (p1 - 1 - i)] = a[i * p2 + j];
    }
  }
}
