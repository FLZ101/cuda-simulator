extern "C" __global__ void mode(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0 || p1 <= 0) {
    return;
  }
  float result = a[0];
  int best_count = 0;
  for (int i = 0; i < p1; ++i) {
    int count = 0;
    for (int j = 0; j < p1; ++j) {
      if (a[j] == a[i]) {
        ++count;
      }
    }
    if (count > best_count) {
      best_count = count;
      result = a[i];
    }
  }
  out[0] = result;
}
