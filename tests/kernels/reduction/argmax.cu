extern "C" __global__ void argmax(const float *in, float *out, int n) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  int best = 0;
  for (int i = 1; i < n; ++i) {
    if (in[i] > in[best]) {
      best = i;
    }
  }
  out[0] = static_cast<float>(best);
}
