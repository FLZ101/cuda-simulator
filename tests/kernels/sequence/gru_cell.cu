extern "C" __global__ void gru_cell(
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
    const float *prev_h = d + i * hidden_size;
    for (int h = 0; h < hidden_size; ++h) {
      float gate_x[3] = {0.0F, 0.0F, 0.0F};
      float gate_h[3] = {0.0F, 0.0F, 0.0F};
      for (int g = 0; g < 3; ++g) {
        for (int k = 0; k < input_size; ++k) {
          gate_x[g] += a[i * input_size + k] *
                       b[(g * hidden_size + h) * input_size + k];
        }
        for (int k = 0; k < hidden_size; ++k) {
          gate_h[g] += prev_h[k] *
                       c[(g * hidden_size + h) * hidden_size + k];
        }
      }
      const float r_gate = 1.0F / (1.0F + __expf(-(gate_x[0] + gate_h[0])));
      const float z_gate = 1.0F / (1.0F + __expf(-(gate_x[1] + gate_h[1])));
      const float n_gate = tanhf(gate_x[2] + r_gate * gate_h[2]);
      out[i * hidden_size + h] =
          (1.0F - z_gate) * n_gate + z_gate * prev_h[h];
    }
  }
}
