extern "C" __global__ void batch_matmul(const float *a, const float *b,
                                        const float *c, const float *d,
                                        const int *idx, float *out, int p0,
                                        int p1, int p2, int p3, int p4,
                                        int p5, int p6, int p7, float f0,
                                        float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int batch = p1;
  const int m = p2;
  const int k = p3;
  const int n = p4;
  for (int z = 0; z < batch; ++z) {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        float sum = 0.0F;
        for (int q = 0; q < k; ++q) {
          sum += a[(z * m + i) * k + q] * b[(z * k + q) * n + j];
        }
        out[(z * m + i) * n + j] = sum;
      }
    }
  }
}
