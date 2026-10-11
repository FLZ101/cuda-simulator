extern "C" __global__ void conv3d(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int in_channels = p1;
  const int depth = p2;
  const int height = p3;
  const int width = p4;
  const int out_channels = p5;
  const int kernel = p6 > 0 ? p6 : 1;
  const int out_d = depth - kernel + 1;
  const int out_h = height - kernel + 1;
  const int out_w = width - kernel + 1;
  for (int oc = 0; oc < out_channels; ++oc) {
    for (int od = 0; od < out_d; ++od) {
      for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
          float value = 0.0F;
          for (int ic = 0; ic < in_channels; ++ic) {
            for (int kd = 0; kd < kernel; ++kd) {
              for (int kh = 0; kh < kernel; ++kh) {
                for (int kw = 0; kw < kernel; ++kw) {
                  const int id = od + kd;
                  const int ih = oh + kh;
                  const int iw = ow + kw;
                  value += a[((ic * depth + id) * height + ih) * width + iw] *
                           b[((((oc * in_channels + ic) * kernel + kd) *
                               kernel + kh) * kernel) + kw];
                }
              }
            }
          }
          out[((oc * out_d + od) * out_h + oh) * out_w + ow] = value;
        }
      }
    }
  }
}
