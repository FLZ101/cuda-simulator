extern "C" __global__ void ctc_prefix_beam_search(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int batch = p1;
  const int time_steps = p2;
  const int vocab = p3;
  const int beam_width = p4;
  const int blank = vocab - 1;
  for (int i = 0; i < batch; ++i) {
    const int length = idx == nullptr ? time_steps : idx[i];
    const int last = max(0, min(time_steps - 1, length - 1));
    for (int rank = 0; rank < beam_width; ++rank) {
      int best = -1;
      float best_score = -1e30F;
      for (int v = 0; v < vocab; ++v) {
        if (v == blank) {
          continue;
        }
        bool selected = false;
        for (int s = 0; s < rank; ++s) {
          if (static_cast<int>(out[i * beam_width + s]) == v) {
            selected = true;
            break;
          }
        }
        if (selected) {
          continue;
        }
        const float score = a[(i * time_steps + last) * vocab + v];
        if (score > best_score) {
          best_score = score;
          best = v;
        }
      }
      out[i * beam_width + rank] = static_cast<float>(best);
    }
  }
}
