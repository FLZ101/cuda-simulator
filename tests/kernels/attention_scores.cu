extern "C" __global__ void attention_scores(const float *query,
                                            const float *key, float *scores,
                                            int batch, int heads, int seq,
                                            int dim) {
  int b = blockIdx.z;
  int h = blockIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch || h >= heads || i >= seq) {
    return;
  }

  float sum = 0.0F;
  for (int j = 0; j < seq; ++j) {
    float dot = 0.0F;
    for (int d = 0; d < dim; ++d) {
      dot += query[((b * heads + h) * seq + i) * dim + d] *
             key[((b * heads + h) * seq + j) * dim + d];
    }
    scores[((b * heads + h) * seq + i) * seq + j] = dot;
    sum += expf(dot);
  }

  for (int j = 0; j < seq; ++j) {
    scores[((b * heads + h) * seq + i) * seq + j] /= sum;
  }
}
