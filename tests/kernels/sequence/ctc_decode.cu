extern "C" __global__ void ctc_decode(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int batch = p1;
  const int time_steps = p2;
  const int vocab = p3;
  const int max_out_len = p4;
  const int blank = p7;
  if (max_out_len <= 0) {
    return;
  }

  for (int i = 0; i < batch; ++i) {
    const int length = idx == nullptr ? time_steps : idx[i];
    int token_count = 0;
    int previous = blank;
    for (int t = 0; t < time_steps && t < length; ++t) {
      int best = 0;
      float best_score = -1e30F;
      for (int v = 0; v < vocab; ++v) {
        const float score = a[(i * time_steps + t) * vocab + v];
        if (score > best_score) {
          best_score = score;
          best = v;
        }
      }
      if (best != blank && best != previous && token_count < max_out_len) {
        out[i * max_out_len + token_count] = static_cast<float>(best);
        ++token_count;
      }
      previous = best;
    }
    for (int j = 0; j < max_out_len; ++j) {
      if (j >= token_count) {
        out[i * max_out_len + j] = -1.0F;
      }
    }
  }
}
