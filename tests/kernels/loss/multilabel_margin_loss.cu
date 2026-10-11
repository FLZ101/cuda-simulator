extern "C" __global__ void multilabel_margin_loss(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int batch = p1;
  const int classes = p2;
  for (int i = 0; i < batch; ++i) {
    const int target = idx == nullptr ? 0 : idx[i];
    const float target_score = a[i * classes + target];
    float loss = 0.0F;
    int count = 0;
    for (int j = 0; j < classes; ++j) {
      if (j == target) {
        continue;
      }
      loss += fmaxf(0.0F, 1.0F - (target_score - a[i * classes + j]));
      ++count;
    }
    out[i] = count > 0 ? loss / static_cast<float>(count) : 0.0F;
  }
}
