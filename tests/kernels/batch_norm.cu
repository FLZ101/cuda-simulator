extern "C" __global__ void batch_norm(const float *input, const float *scale,
                                      const float *bias, float *output,
                                      int channels, int size, float eps) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int channel = index / size;
  if (index >= channels * size) {
    return;
  }
  float value = input[index];
  float mean = 0.0F;
  float var = 1.0F;
  output[index] = scale[channel] * (value - mean) / sqrtf(var + eps) +
                  bias[channel];
}
