extern "C" __global__ void channel_shuffle(const float *in, float *out,
                                           int channels, int groups, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = channels * size;
  if (idx >= total) {
    return;
  }
  int c = idx / size;
  int s = idx % size;
  int g = c % groups;
  int cg = c / groups;
  int src_c = cg * groups + g;
  out[idx] = in[src_c * size + s];
}
