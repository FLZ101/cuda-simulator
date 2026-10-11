extern "C" __global__ void
conv_transpose2d(const float *in, const float *weight, float *out, int channels,
                 int in_h, int in_w, int out_channels, int kernel_h,
                 int kernel_w, int stride, int pad, int out_h, int out_w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = out_channels * out_h * out_w;
  if (idx >= total) {
    return;
  }
  int ow = idx % out_w;
  int oh = (idx / out_w) % out_h;
  int oc = idx / (out_w * out_h);
  float sum = 0.0F;
  for (int ic = 0; ic < channels; ++ic) {
    for (int ky = 0; ky < kernel_h; ++ky) {
      for (int kx = 0; kx < kernel_w; ++kx) {
        int ih = oh + pad - ky;
        int iw = ow + pad - kx;
        if (ih < 0 || iw < 0 || ih >= in_h || iw >= in_w) {
          continue;
        }
        if (ih % stride != 0 || iw % stride != 0) {
          continue;
        }
        int src_h = ih / stride;
        int src_w = iw / stride;
        sum += in[(ic * in_h + src_h) * in_w + src_w] *
               weight[((oc * channels + ic) * kernel_h + ky) * kernel_w + kx];
      }
    }
  }
  out[idx] = sum;
}
