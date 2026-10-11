extern "C" __global__ void pixel_shuffle(const float *in, float *out,
                                         int channels, int upscale, int in_h,
                                         int in_w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int out_c = channels / (upscale * upscale);
  int out_h = in_h * upscale;
  int out_w = in_w * upscale;
  int total = out_c * out_h * out_w;
  if (idx >= total) {
    return;
  }
  int ow = idx % out_w;
  int oh = (idx / out_w) % out_h;
  int oc = idx / (out_w * out_h);
  int x = ow % upscale;
  int y = oh % upscale;
  int w = ow / upscale;
  int h = oh / upscale;
  int ic = oc * upscale * upscale + y * upscale + x;
  out[idx] = in[(ic * in_h + h) * in_w + w];
}
