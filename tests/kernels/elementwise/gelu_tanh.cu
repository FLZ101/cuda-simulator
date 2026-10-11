extern "C" __global__ void gelu_tanh(const float *in, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float x = in[idx];
    float inner = sqrtf(2.0F / 3.14159265358979323846F) *
                  (x + 0.044715F * x * x * x);
    out[idx] = 0.5F * x * (1.0F + tanhf(inner));
  }
}
