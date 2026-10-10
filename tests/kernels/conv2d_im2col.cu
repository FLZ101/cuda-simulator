extern "C" __global__ void conv2d_im2col(const float *input,
                                         const float *weight, float *output,
                                         int c, int h, int w,
                                         int out_c, int kh, int kw,
                                         int out_h, int out_w) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = out_c * out_h * out_w;
  if (index >= total) {
    return;
  }

  int ow = index % out_w;
  int oh = (index / out_w) % out_h;
  int oc = index / (out_w * out_h);
  float sum = 0.0F;

  for (int ic = 0; ic < c; ++ic) {
    for (int ky = 0; ky < kh; ++ky) {
      for (int kx = 0; kx < kw; ++kx) {
        sum += input[(ic * h + oh + ky) * w + ow + kx] *
               weight[((oc * c + ic) * kh + ky) * kw + kx];
      }
    }
  }
  output[index] = sum;
}
