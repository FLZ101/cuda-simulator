extern "C" __global__ void anchor_generate(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int height = p1;
  const int width = p2;
  const float base = static_cast<float>(p3);
  const float stride = f0 > 0.0F ? f0 : 1.0F;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      const int i = h * width + w;
      out[i * 4 + 0] = (static_cast<float>(w) + 0.5F) * stride;
      out[i * 4 + 1] = (static_cast<float>(h) + 0.5F) * stride;
      out[i * 4 + 2] = base;
      out[i * 4 + 3] = base;
    }
  }
}
