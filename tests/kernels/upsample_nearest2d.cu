extern "C" __global__ void upsample_nearest2d(const float *in, float *out,
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
  int iw = ow / scale;
  int ih = oh / scale;
  out[idx] = in[(c * in_h + ih) * in_w + iw];
}
