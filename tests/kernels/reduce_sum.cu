extern "C" __global__ void reduce_sum(const float *in, float *out, int n) {
  extern __shared__ float shared[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  shared[tid] = (i < n) ? in[i] : 0.0F;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[blockIdx.x] = shared[0];
  }
}
