extern "C" __global__ void dropout(const float *in, const float *mask,
                                  float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = in[i] * mask[i];
  }
}
