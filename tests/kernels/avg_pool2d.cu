extern "C" __global__ void avg_pool2d(const float *input, float *output,
                                      int h, int w, int out_h, int out_w,
                                      int kernel) {
  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = blockIdx.y * blockDim.y + threadIdx.y;
  if (ow >= out_w || oh >= out_h) {
    return;
  }
  float sum = 0.0F;
  for (int ky = 0; ky < kernel; ++ky) {
    for (int kx = 0; kx < kernel; ++kx) {
      int ih = oh * kernel + ky;
      int iw = ow * kernel + kx;
      if (ih < h && iw < w) {
        sum += input[ih * w + iw];
      }
    }
  }
  output[oh * out_w + ow] = sum / static_cast<float>(kernel * kernel);
}
