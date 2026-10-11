extern "C" __global__ void transpose(const float *in, float *out, int rows,
                                     int cols) {
  __shared__ float tile[32][32 + 1];
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;

  if (x < cols && y < rows) {
    tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
  }
  __syncthreads();

  int out_x = blockIdx.y * 32 + threadIdx.x;
  int out_y = blockIdx.x * 32 + threadIdx.y;
  if (out_x < rows && out_y < cols) {
    out[out_y * rows + out_x] = tile[threadIdx.x][threadIdx.y];
  }
}
