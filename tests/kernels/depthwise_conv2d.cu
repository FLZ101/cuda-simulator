extern "C" __global__ void
depthwise_conv2d(const float *in, const float *weight, float *out, int channels,
                 int in_h, int in_w, int out_h, int out_w, int kh, int kw) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = channels * out_h * out_w;
  if (idx >= total) {
    return;
  }
  int ow = idx % out_w;
  int oh = (idx / out_w) % out_h;
  int c = idx / (out_w * out_h);
  float sum = 0.0F;
  for (int ky = 0; ky < kh; ++ky) {
    for (int kx = 0; kx < kw; ++kx) {
      int iy = oh + ky;
      int ix = ow + kx;
      if (iy < in_h && ix < in_w) {
        sum +=
            in[(c * in_h + iy) * in_w + ix] * weight[((c * kh + ky) * kw) + kx];
      }
    }
  }
  out[idx] = sum;
}
