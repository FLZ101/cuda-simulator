extern "C" __global__ void matmul_tiled(const float *a, const float *b,
                                        float *c, int m, int n, int k) {
  __shared__ float tile_a[16][16];
  __shared__ float tile_b[16][16];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0F;

  for (int tile = 0; tile < (k + 15) / 16; ++tile) {
    int a_col = tile * 16 + threadIdx.x;
    int b_row = tile * 16 + threadIdx.y;
    tile_a[threadIdx.y][threadIdx.x] =
        (row < m && a_col < k) ? a[row * k + a_col] : 0.0F;
    tile_b[threadIdx.y][threadIdx.x] =
        (b_row < k && col < n) ? b[b_row * n + col] : 0.0F;
    __syncthreads();

#pragma unroll
    for (int i = 0; i < 16; ++i) {
      sum += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < m && col < n) {
    c[row * n + col] = sum;
  }
}
