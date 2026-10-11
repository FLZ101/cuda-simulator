extern "C" __global__ void median(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0 || p1 <= 0) {
    return;
  }
  const int n = p1;
  for (int rank = 0; rank < 2; ++rank) {
    const int target = (n % 2 == 0) ? (n / 2 - 1 + rank) : (n / 2);
    float result = 0.0F;
    bool found = false;
    for (int i = 0; i < n; ++i) {
      int count = 0;
      int duplicates = 0;
      for (int j = 0; j < n; ++j) {
        if (a[j] < a[i]) {
          ++count;
        } else if (a[j] == a[i]) {
          ++duplicates;
        }
      }
      if (count <= target && count + duplicates > target &&
          (!found || a[i] < result)) {
        result = a[i];
        found = true;
      }
    }
    out[0] = found ? result : a[0];
    if (n % 2 != 0) {
      break;
    }
    out[0] = rank == 0 ? (found ? result : a[0]) : (out[0] + (found ? result : a[0])) * 0.5F;
  }
}
