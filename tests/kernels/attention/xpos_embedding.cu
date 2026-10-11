extern "C" __global__ void xpos_embedding(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (int pos = 0; pos < p1; ++pos) {
    for (int dim = 0; dim < p2; dim += 2) {
      const float angle = f0 * static_cast<float>(pos) /
                          powf(10000.0F, static_cast<float>(dim) /
                                              static_cast<float>(p2));
      out[pos * p2 + dim] = __sinf(angle);
      out[pos * p2 + dim + 1] = __cosf(angle);
    }
  }
}
