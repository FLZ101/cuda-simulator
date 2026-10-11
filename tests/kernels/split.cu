extern "C" __global__ void split(const float *in, float *out_a, float *out_b,
                                 int split, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  if (i < split) {
    out_a[i] = in[i];
  } else {
    out_b[i - split] = in[i];
  }
}
