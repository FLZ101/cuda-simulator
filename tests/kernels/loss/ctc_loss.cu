extern "C" __global__ void ctc_loss(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int batch = p1;
  const int time_steps = p2;
  const int vocab = p3;
  const int blank = p7 >= 0 ? p7 : vocab - 1;
  for (int i = 0; i < batch; ++i) {
    const int length = idx == nullptr ? time_steps : idx[i];
    float loss = 0.0F;
    int previous = blank;
    for (int t = 0; t < time_steps && t < length; ++t) {
      int best = 0;
      float max_logit = -1e30F;
      for (int v = 0; v < vocab; ++v) {
        const float value = a[(i * time_steps + t) * vocab + v];
        if (value > max_logit) {
          max_logit = value;
          best = v;
        }
      }
      float sum = 0.0F;
      for (int v = 0; v < vocab; ++v) {
        sum += __expf(a[(i * time_steps + t) * vocab + v] - max_logit);
      }
      const float log_prob =
          a[(i * time_steps + t) * vocab + best] - max_logit - __logf(sum);
      if (best != blank && best != previous) {
        loss -= log_prob;
      }
      previous = best;
    }
    out[i] = loss;
  }
}
