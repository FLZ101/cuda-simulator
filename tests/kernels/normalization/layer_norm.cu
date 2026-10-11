extern "C" __global__ void layer_norm(const float *input, float *output,
                                      int n, int hidden, float eps) {
  int row = blockIdx.x;
  if (row >= n) {
    return;
  }
  float mean = 0.0F;
  float var = 0.0F;
  for (int i = 0; i < hidden; ++i) {
    float value = input[row * hidden + i];
    mean += value;
    var += value * value;
  }
  mean /= static_cast<float>(hidden);
  var = var / static_cast<float>(hidden) - mean * mean;
  for (int i = 0; i < hidden; ++i) {
    output[row * hidden + i] =
        (input[row * hidden + i] - mean) / sqrtf(var + eps);
  }
}
