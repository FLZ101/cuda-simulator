extern "C" __global__ void deform_conv2d(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int height = p1;
  const int width = p2;
  const int kernel = p3;
  const int out_h = p4;
  const int out_w = p5;
  for (int oh = 0; oh < out_h; ++oh) {
    for (int ow = 0; ow < out_w; ++ow) {
      float value = 0.0F;
      for (int kh = 0; kh < kernel; ++kh) {
        for (int kw = 0; kw < kernel; ++kw) {
          const int off =
              (((oh * out_w + ow) * kernel + kh) * kernel + kw) * 2;
          const float sy = oh + c[off];
          const float sx = ow + c[off + 1];
          int y = static_cast<int>(sy + 0.5F);
          int x = static_cast<int>(sx + 0.5F);
          y = max(0, min(height - 1, y));
          x = max(0, min(width - 1, x));
          value += a[y * width + x] * b[kh * kernel + kw];
        }
      }
      out[oh * out_w + ow] = value;
    }
  }
}
