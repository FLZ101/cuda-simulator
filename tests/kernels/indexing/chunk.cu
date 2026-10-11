extern "C" __global__ void chunk(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int chunks = p1;
  const int length = p2;
  const int chunk_size = (length + chunks - 1) / chunks;
  for (int ch = 0; ch < chunks; ++ch) {
    const int src_chunk = chunks - 1 - ch;
    for (int j = 0; j < chunk_size; ++j) {
      const int src = src_chunk * chunk_size + j;
      const int dst = ch * chunk_size + j;
      if (dst < length && src < length) {
        out[dst] = a[src];
      }
    }
  }
}
