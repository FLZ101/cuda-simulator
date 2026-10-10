extern "C" __global__ void gelu(const float *input, float *output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = input[i];
    output[i] = 0.5F * x * (1.0F + tanhf(0.7978845608F *
                                          (x + 0.044715F * x * x * x)));
  }
}
