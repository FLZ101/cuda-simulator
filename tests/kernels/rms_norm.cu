extern "C" __global__ void rms_norm(const float *input, float *output,
                                    int rows, int hidden, float eps) {
  int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  float square_sum = 0.0F;
  for (int i = 0; i < hidden; ++i) {
    float value = input[row * hidden + i];
    square_sum += value * value;
  }
  float rms = sqrtf(square_sum / static_cast<float>(hidden));
  for (int i = 0; i < hidden; ++i) {
    output[row * hidden + i] =
        input[row * hidden + i] / (rms + eps);
  }
}
