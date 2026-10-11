extern "C" __global__ void generate_proposals(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int box_count = p1;
  const int top_k = p2;
  for (int rank = 0; rank < top_k; ++rank) {
    int best = -1;
    float best_score = -1e30F;
    for (int i = 0; i < box_count; ++i) {
      bool selected = false;
      for (int s = 0; s < rank; ++s) {
        if (static_cast<int>(out[s]) == i) {
          selected = true;
          break;
        }
      }
      if (!selected && a[i] > best_score) {
        best = i;
        best_score = a[i];
      }
    }
    out[rank] = best < 0 ? -1.0F : static_cast<float>(best);
  }
  for (int rank = 0; rank < top_k; ++rank) {
    const int index = static_cast<int>(out[rank]);
    for (int j = 0; j < 4; ++j) {
      out[rank * 4 + j] = index >= 0 ? b[index * 4 + j] : 0.0F;
    }
  }
}
