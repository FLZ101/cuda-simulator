extern "C" __global__ void kv_cache_update(const float *a, const float *b,
                                          const float *c, const float *d,
                                          const int *idx, float *out, int p0,
                                          int p1, int p2, int p3, int p4,
                                          int p5, int p6, int p7, float f0,
                                          float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int cache_len = p1;
  const int dim = p2;
  for (int i = 0; i < cache_len; ++i) {
    for (int e = 0; e < dim; ++e) {
      out[i * dim + e] = a[i * dim + e];
    }
  }
  const int pos = static_cast<int>(f0);
  for (int e = 0; e < dim; ++e) {
    out[pos * dim + e] = b[e];
  }
}
