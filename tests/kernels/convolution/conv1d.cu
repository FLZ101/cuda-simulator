extern "C" __global__ void conv1d(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int in_channels = p1;
  const int length = p2;
  const int out_channels = p3;
  const int kernel = p4;
  const int stride = p5 > 0 ? p5 : 1;
  const int padding = p6;
  const int dilation = p7 > 0 ? p7 : 1;
  const int out_length =
      (length + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
  for (int oc = 0; oc < out_channels; ++oc) {
    for (int ol = 0; ol < out_length; ++ol) {
      float value = 0.0F;
      for (int ic = 0; ic < in_channels; ++ic) {
        for (int k = 0; k < kernel; ++k) {
          const int il = ol * stride - padding + k * dilation;
          if (il >= 0 && il < length) {
            value += a[ic * length + il] *
                     b[((oc * in_channels + ic) * kernel) + k];
          }
        }
      }
      out[oc * out_length + ol] = value;
    }
  }
}
