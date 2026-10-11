extern "C" __global__ void batched_nms(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int batch = p1;
  const int boxes_per_batch = p2;
  for (int z = 0; z < batch; ++z) {
    for (int i = 0; i < boxes_per_batch; ++i) {
      out[z * boxes_per_batch + i] = 0.0F;
    }
    for (int rank = 0; rank < boxes_per_batch; ++rank) {
      int best = -1;
      float best_score = -1e30F;
      for (int i = 0; i < boxes_per_batch; ++i) {
        const int index = z * boxes_per_batch + i;
        if (out[index] == 0.0F && b[index] > best_score) {
          best = i;
          best_score = b[index];
        }
      }
      if (best < 0) {
        break;
      }
      out[z * boxes_per_batch + best] = best_score;
      const float *box_best = a + (z * boxes_per_batch + best) * 4;
      for (int j = 0; j < boxes_per_batch; ++j) {
        const int index = z * boxes_per_batch + j;
        if (out[index] != 0.0F) {
          continue;
        }
        const float *box_j = a + (z * boxes_per_batch + j) * 4;
        const float xx1 = fmaxf(box_best[0], box_j[0]);
        const float yy1 = fmaxf(box_best[1], box_j[1]);
        const float xx2 = fminf(box_best[2], box_j[2]);
        const float yy2 = fminf(box_best[3], box_j[3]);
        const float inter = fmaxf(0.0F, xx2 - xx1) * fmaxf(0.0F, yy2 - yy1);
        const float area1 = fmaxf(0.0F, box_best[2] - box_best[0]) *
                            fmaxf(0.0F, box_best[3] - box_best[1]);
        const float area2 = fmaxf(0.0F, box_j[2] - box_j[0]) *
                            fmaxf(0.0F, box_j[3] - box_j[1]);
        const float iou = inter / (area1 + area2 - inter + 1e-6F);
        if (iou > f0) {
          out[index] = -1.0F;
        }
      }
    }
  }
}
