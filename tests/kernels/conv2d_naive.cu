extern "C" __global__ void conv2d_naive(const float *input,
                                        const float *weight, float *output,
                                        int n, int c, int h, int w,
                                        int out_c, int kh, int kw,
                                        int out_h, int out_w) {
  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = blockIdx.y * blockDim.y + threadIdx.y;
  int oc = blockIdx.z;
  int batch = 0;

  if (ow < out_w && oh < out_h && oc < out_c) {
    float sum = 0.0F;
    for (int ic = 0; ic < c; ++ic) {
      for (int ky = 0; ky < kh; ++ky) {
        for (int kx = 0; kx < kw; ++kx) {
          int ih = oh + ky;
          int iw = ow + kx;
          sum += input[((batch * c + ic) * h + ih) * w + iw] *
                 weight[((oc * c + ic) * kh + ky) * kw + kx];
        }
      }
    }
    output[(batch * out_c + oc) * out_h * out_w + oh * out_w + ow] = sum;
  }
}
