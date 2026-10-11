extern "C" __global__ void roi_align(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int channels = p1;
  const int height = p2;
  const int width = p3;
  const int boxes = p4;
  const int pool_h = p5;
  const int pool_w = p6;
  if (channels <= 0 || height <= 0 || width <= 0 || boxes <= 0 ||
      pool_h <= 0 || pool_w <= 0) {
    return;
  }

  for (int n = 0; n < boxes; ++n) {
    float y1 = b[n * 4 + 0] * static_cast<float>(height);
    float x1 = b[n * 4 + 1] * static_cast<float>(width);
    float y2 = b[n * 4 + 2] * static_cast<float>(height);
    float x2 = b[n * 4 + 3] * static_cast<float>(width);
    const float roi_h = fmaxf(y2 - y1, 1e-3F);
    const float roi_w = fmaxf(x2 - x1, 1e-3F);
    for (int ch = 0; ch < channels; ++ch) {
      for (int ph = 0; ph < pool_h; ++ph) {
        const float y = y1 + (static_cast<float>(ph) + 0.5F) * roi_h /
                                 static_cast<float>(pool_h);
        for (int pw = 0; pw < pool_w; ++pw) {
          const float x =
              x1 + (static_cast<float>(pw) + 0.5F) * roi_w /
                       static_cast<float>(pool_w);
          int y0 = static_cast<int>(y);
          int x0 = static_cast<int>(x);
          y0 = max(0, min(height - 1, y0));
          x0 = max(0, min(width - 1, x0));
          int y1i = min(height - 1, y0 + 1);
          int x1i = min(width - 1, x0 + 1);
          const float ly = y - static_cast<float>(y0);
          const float lx = x - static_cast<float>(x0);
          const float v00 = a[(ch * height + y0) * width + x0];
          const float v01 = a[(ch * height + y0) * width + x1i];
          const float v10 = a[(ch * height + y1i) * width + x0];
          const float v11 = a[(ch * height + y1i) * width + x1i];
          const float top = v00 * (1.0F - lx) + v01 * lx;
          const float bottom = v10 * (1.0F - lx) + v11 * lx;
          out[((n * channels + ch) * pool_h + ph) * pool_w + pw] =
              top * (1.0F - ly) + bottom * ly;
        }
      }
    }
  }
}
