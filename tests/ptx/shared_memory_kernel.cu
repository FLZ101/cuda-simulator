extern "C" __global__ void shared_sum(const float *in, float *out) {
  __shared__ float tile[32];
  const int tid = threadIdx.x;
  tile[tid] = in[tid];
  __syncthreads();

  if (tid == 0) {
    float sum = 0.0F;
    for (int i = 0; i < 32; ++i) {
      sum += tile[i];
    }
    out[0] = sum;
  }
}
