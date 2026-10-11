extern "C" __global__ void unfold(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int channels = p1;
  const int height = p2;
  const int width = p3;
  const int kernel = p4;
  const int out_h = height - kernel + 1;
  const int out_w = width - kernel + 1;
  for (int ch = 0; ch < channels; ++ch) {
    for (int oh = 0; oh < out_h; ++oh) {
      for (int ow = 0; ow < out_w; ++ow) {
        for (int kh = 0; kh < kernel; ++kh) {
          for (int kw = 0; kw < kernel; ++kw) {
            out[(((ch * out_h + oh) * out_w + ow) * kernel + kh) * kernel +
                kw] = a[(ch * height + oh + kh) * width + ow + kw];
          }
        }
      }
    }
  }
}
