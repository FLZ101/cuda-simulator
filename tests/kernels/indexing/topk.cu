extern "C" __global__ void topk(const float *a, const float *b,
                                const float *c, const float *d,
                                const int *idx, float *out, int p0, int p1,
                                int p2, int p3, int p4, int p5, int p6,
                                int p7, float f0, float f1, float f2,
                                float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (int i = 0; i < p2; ++i) {
    out[i] = a[i];
  }
  for (int i = 0; i < p2; ++i) {
    for (int j = i + 1; j < p2; ++j) {
      if (out[j] > out[i]) {
        const float tmp = out[i];
        out[i] = out[j];
        out[j] = tmp;
      }
    }
  }
  for (int i = 0; i < p1; ++i) {
    for (int j = 0; j < p2; ++j) {
      if (a[i] > out[p2 - 1] && (i >= p2)) {
        out[p2 - 1] = a[i];
        for (int q = p2 - 1; q > 0; --q) {
          if (out[q] > out[q - 1]) {
            const float tmp = out[q];
            out[q] = out[q - 1];
            out[q - 1] = tmp;
          }
        }
      }
    }
  }
}
