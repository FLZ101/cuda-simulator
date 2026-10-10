extern "C" __global__ void softmax_row(const float *in, float *out, int rows,
                                       int cols) {
  int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  float max_value = in[row * cols];
  for (int i = 1; i < cols; ++i) {
    max_value = fmaxf(max_value, in[row * cols + i]);
  }

  float sum = 0.0F;
  for (int i = 0; i < cols; ++i) {
    float value = expf(in[row * cols + i] - max_value);
    out[row * cols + i] = value;
    sum += value;
  }

  for (int i = 0; i < cols; ++i) {
    out[row * cols + i] /= sum;
  }
}
