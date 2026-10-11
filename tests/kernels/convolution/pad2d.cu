extern "C" __global__ void pad2d(const float *in, float *out, int channels,
                                 int in_h, int in_w, int pad_h, int pad_w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int out_h = in_h + 2 * pad_h;
  int out_w = in_w + 2 * pad_w;
  int total = channels * out_h * out_w;
  if (idx >= total) {
    return;
  }
  int ow = idx % out_w;
  int oh = (idx / out_w) % out_h;
  int c = idx / (out_w * out_h);
  int iw = ow - pad_w;
  int ih = oh - pad_h;
  out[idx] = (iw >= 0 && iw < in_w && ih >= 0 && ih < in_h)
                 ? in[(c * in_h + ih) * in_w + iw]
                 : 0.0F;
}
