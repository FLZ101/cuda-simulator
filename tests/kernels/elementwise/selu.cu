extern "C" __global__ void selu(const float *in, float *out, int n) {
  const float alpha = 1.6732632423543772F;
  const float scale = 1.0507009873554805F;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float x = in[idx];
    out[idx] = scale * (x > 0.0F ? x : alpha * (expf(x) - 1.0F));
  }
}
