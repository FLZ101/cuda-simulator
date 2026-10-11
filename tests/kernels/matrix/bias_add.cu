extern "C" __global__ void bias_add(const float *input, const float *bias,
                                    float *output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = input[i] + bias[blockIdx.y];
  }
}
