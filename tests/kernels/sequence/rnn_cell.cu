extern "C" __global__ void rnn_cell(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int batch = p1;
  const int input_size = p2;
  const int hidden_size = p3;
  for (int i = 0; i < batch; ++i) {
    for (int h = 0; h < hidden_size; ++h) {
      float value = 0.0F;
      for (int k = 0; k < input_size; ++k) {
        value += a[i * input_size + k] * b[k * hidden_size + h];
      }
      for (int k = 0; k < hidden_size; ++k) {
        value += d[i * hidden_size + k] * c[k * hidden_size + h];
      }
      out[i * hidden_size + h] = f0 == 0.0F ? tanhf(value) : fmaxf(0.0F, value);
    }
  }
}
