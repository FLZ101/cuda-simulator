extern "C" __global__ void local_attention(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int batch = p1;
  const int heads = p2;
  const int seq = p3;
  const int dim = p4;
  const int window = p5;
  const float scale = rsqrtf(static_cast<float>(dim));
  for (int z = 0; z < batch; ++z) {
    for (int h = 0; h < heads; ++h) {
      for (int i = 0; i < seq; ++i) {
        float max_score = -1e30F;
        for (int j = 0; j < seq; ++j) {
          if (abs(i - j) > window) {
            continue;
          }
          float score = 0.0F;
          for (int e = 0; e < dim; ++e) {
            score += a[((z * heads + h) * seq + i) * dim + e] *
                     b[((z * heads + h) * seq + j) * dim + e];
          }
          max_score = fmaxf(max_score, score * scale);
        }
        float sum = 0.0F;
        for (int e = 0; e < dim; ++e) {
          out[((z * heads + h) * seq + i) * dim + e] = 0.0F;
        }
        for (int j = 0; j < seq; ++j) {
          if (abs(i - j) > window) {
            continue;
          }
          float score = 0.0F;
          for (int e = 0; e < dim; ++e) {
            score += a[((z * heads + h) * seq + i) * dim + e] *
                     b[((z * heads + h) * seq + j) * dim + e];
          }
          score = __expf(score * scale - max_score);
          sum += score;
          for (int e = 0; e < dim; ++e) {
            out[((z * heads + h) * seq + i) * dim + e] +=
                score * c[((z * heads + h) * seq + j) * dim + e];
          }
        }
        for (int e = 0; e < dim; ++e) {
          out[((z * heads + h) * seq + i) * dim + e] /= sum;
        }
      }
    }
  }
}
