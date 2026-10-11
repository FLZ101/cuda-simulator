extern "C" __global__ void kron(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int m1 = p1;
  const int n1 = p2;
  const int m2 = p3;
  const int n2 = p4;
  for (int i = 0; i < m1; ++i) {
    for (int j = 0; j < n1; ++j) {
      for (int k = 0; k < m2; ++k) {
        for (int l = 0; l < n2; ++l) {
          out[((i * m2 + k) * (n1 * n2)) + (j * n2 + l)] =
              a[i * n1 + j] * b[k * n2 + l];
        }
      }
    }
  }
}
