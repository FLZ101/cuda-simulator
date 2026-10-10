extern "C" __global__ void matmul_naive(const float *a, const float *b,
                                        float *c, int m, int n, int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m && col < n) {
    float sum = 0.0F;
    for (int i = 0; i < k; ++i) {
      sum += a[row * k + i] * b[i * n + col];
    }
    c[row * n + col] = sum;
  }
}
