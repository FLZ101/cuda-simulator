extern "C" __global__ void elu(const float *in, float *out, float alpha,
                               int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = in[i] >= 0.0F ? in[i] : alpha * (expf(in[i]) - 1.0F);
  }
}
