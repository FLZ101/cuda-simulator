extern "C" __global__ void masked_softmax(const float *a, const float *b,
                                          const float *c, const float *d,
                                          const int *idx, float *out, int p0,
                                          int p1, int p2, int p3, int p4,
                                          int p5, int p6, int p7, float f0,
                                          float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (int row = 0; row < p1; ++row) {
    float max_value = -1e30F;
    for (int col = 0; col < p2; ++col) {
      if (b[row * p2 + col] != 0.0F) {
        max_value = fmaxf(max_value, a[row * p2 + col]);
      }
    }
    float sum = 0.0F;
    for (int col = 0; col < p2; ++col) {
      if (b[row * p2 + col] != 0.0F) {
        out[row * p2 + col] = __expf(a[row * p2 + col] - max_value);
        sum += out[row * p2 + col];
      } else {
        out[row * p2 + col] = 0.0F;
      }
    }
    for (int col = 0; col < p2; ++col) {
      if (b[row * p2 + col] != 0.0F) {
        out[row * p2 + col] /= sum;
      }
    }
  }
}
