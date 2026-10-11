extern "C" __global__ void fold(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int channels = p1;
  const int blocks = p2;
  const int kernel = p3;
  const int height = p4;
  const int width = p5;
  for (int ch = 0; ch < channels; ++ch) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        float value = 0.0F;
        const int block = h * width + w;
        for (int k = 0; k < kernel * kernel; ++k) {
          value += a[(ch * blocks + block) * (kernel * kernel) + k];
        }
        out[(ch * height + h) * width + w] = value;
      }
    }
  }
}
