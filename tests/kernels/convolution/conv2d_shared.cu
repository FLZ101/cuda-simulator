extern "C" __global__ void conv2d_shared(const float *input,
                                         const float *weight, float *output,
                                         int c, int h, int w,
                                         int out_c, int kh, int kw,
                                         int out_h, int out_w) {
  __shared__ float tile[18][18];
  int ow = blockIdx.x * 16 + threadIdx.x;
  int oh = blockIdx.y * 16 + threadIdx.y;

  for (int oc = 0; oc < out_c; ++oc) {
    float sum = 0.0F;
    for (int ic = 0; ic < c; ++ic) {
      int ih = oh + threadIdx.y;
      int iw = ow + threadIdx.x;
      tile[threadIdx.y][threadIdx.x] =
          (ih < h && iw < w) ? input[(ic * h + ih) * w + iw] : 0.0F;
      __syncthreads();
      if (threadIdx.y < kh && threadIdx.x < kw && oh < out_h && ow < out_w) {
        sum += tile[threadIdx.y][threadIdx.x] *
               weight[((oc * c + ic) * kh + threadIdx.y) * kw + threadIdx.x];
      }
      __syncthreads();
    }
    if (ow < out_w && oh < out_h) {
      output[(oc * out_h + oh) * out_w + ow] = sum;
    }
  }
}
