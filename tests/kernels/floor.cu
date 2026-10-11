extern "C" __global__ void floor_kernel(const float *in, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = floorf(in[i]);
  }
}
