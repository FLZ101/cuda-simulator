extern "C" __global__ void cumsum(const float *in, float *out, int rows,
                                  int cols) {
  int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  float running = 0.0F;
  for (int i = 0; i < cols; ++i) {
    running += in[row * cols + i];
    out[row * cols + i] = running;
  }
}
