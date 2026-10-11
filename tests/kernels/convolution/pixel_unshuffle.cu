extern "C" __global__ void pixel_unshuffle(const float *in, float *out,
                                           int channels, int downscale,
                                           int in_h, int in_w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int out_channels = channels * downscale * downscale;
  int out_h = in_h / downscale;
  int out_w = in_w / downscale;
  int total = out_channels * out_h * out_w;
  if (idx >= total) {
    return;
  }
  int ow = idx % out_w;
  int oh = (idx / out_w) % out_h;
  int oc = idx / (out_w * out_h);
  int ic = oc % channels;
  int rem = oc / channels;
  int x = rem % downscale;
  int y = rem / downscale;
  int ih = oh * downscale + y;
  int iw = ow * downscale + x;
  out[idx] = in[(ic * in_h + ih) * in_w + iw];
}
