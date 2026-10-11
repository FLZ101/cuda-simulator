extern "C" __global__ void flash_attention_simt(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (int i = 0; i < p0; ++i) {
    out[i] = 0.0F;
  }
  const float scale = rsqrtf(static_cast<float>(p4));
  for (int z = 0; z < p1; ++z) {
    for (int h = 0; h < p2; ++h) {
      for (int i = 0; i < p3; ++i) {
        float max_value = -1e30F;
        for (int j = 0; j < p3; ++j) {
          float score = 0.0F;
          for (int e = 0; e < p4; ++e) {
            score += a[((z * p2 + h) * p3 + i) * p4 + e] *
                     b[((z * p2 + h) * p3 + j) * p4 + e];
          }
          max_value = fmaxf(max_value, score * scale);
        }
        float sum = 0.0F;
        for (int j = 0; j < p3; ++j) {
          float score = 0.0F;
          for (int e = 0; e < p4; ++e) {
            score += a[((z * p2 + h) * p3 + i) * p4 + e] *
                     b[((z * p2 + h) * p3 + j) * p4 + e];
          }
          score = __expf(score * scale - max_value);
          sum += score;
          for (int e = 0; e < p4; ++e) {
            out[((z * p2 + h) * p3 + i) * p4 + e] +=
                score * c[((z * p2 + h) * p3 + j) * p4 + e];
          }
        }
        for (int e = 0; e < p4; ++e) {
          out[((z * p2 + h) * p3 + i) * p4 + e] /= sum;
        }
      }
    }
  }
}
