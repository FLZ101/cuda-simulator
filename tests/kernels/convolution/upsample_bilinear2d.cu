extern "C" __global__ void upsample_bilinear2d(const float *in, float *out,
                                               int channels, int in_h, int in_w,
                                               int scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int out_h = in_h * scale;
  int out_w = in_w * scale;
  int total = channels * out_h * out_w;
  if (idx >= total) {
    return;
  }
  int ow = idx % out_w;
  int oh = (idx / out_w) % out_h;
  int c = idx / (out_w * out_h);
  float src_h =
      (static_cast<float>(oh) + 0.5F) / static_cast<float>(scale) - 0.5F;
  float src_w =
      (static_cast<float>(ow) + 0.5F) / static_cast<float>(scale) - 0.5F;
  int h0 = static_cast<int>(floorf(src_h));
  int w0 = static_cast<int>(floorf(src_w));
  int h1 = min(h0 + 1, in_h - 1);
  int w1 = min(w0 + 1, in_w - 1);
  h0 = max(h0, 0);
  w0 = max(w0, 0);
  float lh = src_h - static_cast<float>(h0);
  float lw = src_w - static_cast<float>(w0);
  float top = in[(c * in_h + h0) * in_w + w0] * (1.0F - lw) +
              in[(c * in_h + h0) * in_w + w1] * lw;
  float bottom = in[(c * in_h + h1) * in_w + w0] * (1.0F - lw) +
                 in[(c * in_h + h1) * in_w + w1] * lw;
  out[idx] = top * (1.0F - lh) + bottom * lh;
}
