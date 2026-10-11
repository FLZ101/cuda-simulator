extern "C" __global__ void adaptive_avg_pool2d(const float *in, float *out,
                                               int channels, int in_h, int in_w,
                                               int out_h, int out_w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = channels * out_h * out_w;
  if (idx >= total) {
    return;
  }
  int ow = idx % out_w;
  int oh = (idx / out_w) % out_h;
  int c = idx / (out_w * out_h);
  int h0 = (oh * in_h) / out_h;
  int h1 = ((oh + 1) * in_h + out_h - 1) / out_h;
  int w0 = (ow * in_w) / out_w;
  int w1 = ((ow + 1) * in_w + out_w - 1) / out_w;
  float sum = 0.0F;
  for (int h = h0; h < h1; ++h) {
    for (int w = w0; w < w1; ++w) {
      sum += in[(c * in_h + h) * in_w + w];
    }
  }
  out[idx] = sum / static_cast<float>((h1 - h0) * (w1 - w0));
}
