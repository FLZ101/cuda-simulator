extern "C" __global__ void causal_mask(float *scores, int seq_len) {
  int row = blockIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < seq_len && col < seq_len && col > row) {
    scores[row * seq_len + col] = -1.0e30F;
  }
}
