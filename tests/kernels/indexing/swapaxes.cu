extern "C" __global__ void swapaxes(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int rows = p1;
  const int cols = p2;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      out[j * rows + i] = a[i * cols + j];
    }
  }
}
