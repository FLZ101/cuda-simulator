extern "C" __global__ void cross_entropy(const float *logits,
                                         const int *targets, float *out,
                                         int batch, int classes) {
  int b = blockIdx.x;
  if (b >= batch) {
    return;
  }
  int target = targets[b];
  float max_value = logits[b * classes];
  for (int i = 1; i < classes; ++i) {
    max_value = fmaxf(max_value, logits[b * classes + i]);
  }
  float sum = 0.0F;
  for (int i = 0; i < classes; ++i) {
    sum += expf(logits[b * classes + i] - max_value);
  }
  out[b] = -(logits[b * classes + target] - max_value - logf(sum));
}
