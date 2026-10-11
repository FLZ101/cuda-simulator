extern "C" __global__ void nms(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  if (p1 <= 0 || p2 != 4) {
    return;
  }

  for (int i = 0; i < p1; ++i) {
    out[i] = 0.0F;
  }

  for (int rank = 0; rank < p1; ++rank) {
    int best = -1;
    float best_score = -1e30F;
    for (int i = 0; i < p1; ++i) {
      const float score = b == nullptr ? 0.0F : b[i];
      if (out[i] == 0.0F && score > best_score) {
        best = i;
        best_score = score;
      }
    }
    if (best < 0) {
      break;
    }
    out[best] = best_score;
    const float *box_best = a + best * 4;
    for (int j = 0; j < p1; ++j) {
      if (out[j] != 0.0F) {
        continue;
      }
      const float *box_j = a + j * 4;
      const float xx1 = fmaxf(box_best[0], box_j[0]);
      const float yy1 = fmaxf(box_best[1], box_j[1]);
      const float xx2 = fminf(box_best[2], box_j[2]);
      const float yy2 = fminf(box_best[3], box_j[3]);
      const float w = fmaxf(0.0F, xx2 - xx1);
      const float h = fmaxf(0.0F, yy2 - yy1);
      const float inter = w * h;
      const float area1 =
          fmaxf(0.0F, box_best[2] - box_best[0]) *
          fmaxf(0.0F, box_best[3] - box_best[1]);
      const float area2 =
          fmaxf(0.0F, box_j[2] - box_j[0]) *
          fmaxf(0.0F, box_j[3] - box_j[1]);
      const float iou = inter / (area1 + area2 - inter + 1e-6F);
      if (iou > f0) {
        out[j] = -1.0F;
      }
    }
  }
}
