extern "C" __global__ void transformer_decoder(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int batch = p1;
  const int seq = p2;
  const int dim = p3;
  const float scale = rsqrtf(static_cast<float>(dim));
  for (int i = 0; i < batch; ++i) {
    for (int t = 0; t < seq; ++t) {
      float max_score = -1e30F;
      for (int j = 0; j < seq; ++j) {
        float score = 0.0F;
        for (int e = 0; e < dim; ++e) {
          score += a[(i * seq + t) * dim + e] * a[(i * seq + j) * dim + e];
        }
        max_score = fmaxf(max_score, score * scale);
      }
      float sum = 0.0F;
      for (int e = 0; e < dim; ++e) {
        out[(i * seq + t) * dim + e] = 0.0F;
      }
      for (int j = 0; j < seq; ++j) {
        float score = 0.0F;
        for (int e = 0; e < dim; ++e) {
          score += a[(i * seq + t) * dim + e] * a[(i * seq + j) * dim + e];
        }
        score = __expf(score * scale - max_score);
        sum += score;
        for (int e = 0; e < dim; ++e) {
          out[(i * seq + t) * dim + e] +=
              score * a[(i * seq + j) * dim + e];
        }
      }
      for (int e = 0; e < dim; ++e) {
        out[(i * seq + t) * dim + e] =
            out[(i * seq + t) * dim + e] / sum +
            a[(i * seq + t) * dim + e];
      }
    }
  }
}
