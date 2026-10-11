extern "C" __global__ void reflection_pad2d(const float *in, float *out,
                                            int channels, int in_h, int in_w,
                                            int pad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int out_h = in_h + 2 * pad;
  int out_w = in_w + 2 * pad;
  int total = channels * out_h * out_w;
  if (idx >= total) {
    return;
  }
  int ow = idx % out_w;
  int oh = (idx / out_w) % out_h;
  int c = idx / (out_w * out_h);
  int iw = ow - pad;
  int ih = oh - pad;
  iw = iw < 0 ? -iw - 1 : (iw >= in_w ? 2 * in_w - iw - 1 : iw);
  ih = ih < 0 ? -ih - 1 : (ih >= in_h ? 2 * in_h - ih - 1 : ih);
  out[idx] = in[(c * in_h + ih) * in_w + iw];
}
