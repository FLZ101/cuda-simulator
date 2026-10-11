extern "C" __global__ void celu(const float *in, float *out, int n,
                                float alpha) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx] > 0.0F ? in[idx]
                              : alpha * (expf(in[idx] / alpha) - 1.0F);
  }
}
