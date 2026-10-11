extern "C" __global__ void local_response_norm(const float *in, float *out,
                                               int channels, int size, int n,
                                               float alpha, float beta,
                                               float k) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = channels * size;
  if (idx >= total) {
    return;
  }
  int c = idx / size;
  int s = idx % size;
  int half = n / 2;
  float sum = 0.0F;
  for (int j = -half; j <= half; ++j) {
    int cc = c + j;
    if (cc >= 0 && cc < channels) {
      sum += in[cc * size + s] * in[cc * size + s];
    }
  }
  out[idx] = in[idx] / powf(k + alpha * sum, beta);
}
