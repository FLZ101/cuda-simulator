extern "C" __global__ void beam_search(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int batch = p1;
  const int vocab = p2;
  const int beam_width = p3;
  if (beam_width <= 0) {
    return;
  }

  for (int i = 0; i < batch; ++i) {
    for (int k = 0; k < beam_width; ++k) {
      out[i * beam_width + k] = -1.0F;
    }
    for (int k = 0; k < beam_width; ++k) {
      int best_idx = -1;
      float best_score = -1e30F;
      for (int v = 0; v < vocab; ++v) {
        bool already_selected = false;
        for (int s = 0; s < k; ++s) {
          if (static_cast<int>(out[i * beam_width + s]) == v) {
            already_selected = true;
            break;
          }
        }
        if (already_selected) {
          continue;
        }
        const float score = a[i * vocab + v];
        if (score > best_score) {
          best_score = score;
          best_idx = v;
        }
      }
      out[i * beam_width + k] = static_cast<float>(best_idx);
    }
  }
}
