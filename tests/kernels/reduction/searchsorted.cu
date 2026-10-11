extern "C" __global__ void searchsorted(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int n = p1;
  const int queries = p2;
  for (int q = 0; q < queries; ++q) {
    const float value = b[q];
    int lo = 0;
    int hi = n;
    while (lo < hi) {
      const int mid = (lo + hi) / 2;
      if (a[mid] < value) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    out[q] = static_cast<float>(lo);
  }
}
