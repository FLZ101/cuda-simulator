extern "C" __global__ void fused_add_norm(const float *in,
                                          const float *residual, float *out,
                                          int rows, int hidden, float eps) {
  int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  float mean = 0.0F;
  float var = 0.0F;
  for (int i = 0; i < hidden; ++i) {
    float v = in[row * hidden + i];
    mean += v;
    var += v * v;
  }
  mean /= static_cast<float>(hidden);
  var = var / static_cast<float>(hidden) - mean * mean;
  for (int i = 0; i < hidden; ++i) {
    out[row * hidden + i] = (in[row * hidden + i] - mean) / sqrtf(var + eps) +
                            residual[row * hidden + i];
  }
}
