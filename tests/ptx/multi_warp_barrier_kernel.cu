extern "C" __global__ void barrier_copy(const float *in, float *out) {
  __shared__ float tile[64];
  const int tid = threadIdx.x;
  tile[tid] = in[tid];
  __syncthreads();
  out[tid] = tile[tid];
}
