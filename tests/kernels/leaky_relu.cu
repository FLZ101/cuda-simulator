extern "C" __global__ void leaky_relu(const float *in, float *out, float alpha,
                                      int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = in[i] >= 0.0F ? in[i] : alpha * in[i];
  }
}
