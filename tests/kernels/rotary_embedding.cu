extern "C" __global__ void rotary_embedding(const float *in,
                                            const float *cos_table,
                                            const float *sin_table, float *out,
                                            int tokens, int dim) {
  int token = blockIdx.x;
  if (token >= tokens) {
    return;
  }
  for (int d = 0; d < dim; d += 2) {
    int pair = d / 2;
    int table_idx = token * (dim / 2) + pair;
    float c = cos_table[table_idx];
    float s = sin_table[table_idx];
    float x0 = in[token * dim + d];
    float x1 = in[token * dim + d + 1];
    out[token * dim + d] = x0 * c - x1 * s;
    out[token * dim + d + 1] = x0 * s + x1 * c;
  }
}
