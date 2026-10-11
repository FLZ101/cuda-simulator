extern "C" __global__ void repeat_kv(const float *in, float *out, int batch,
                                     int kv_heads, int q_heads, int seq,
                                     int dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * q_heads * seq * dim;
  if (idx >= total) {
    return;
  }
  int d = idx % dim;
  int s = (idx / dim) % seq;
  int qh = (idx / (dim * seq)) % q_heads;
  int b = idx / (dim * seq * q_heads);
  int ratio = q_heads / kv_heads;
  int kh = qh / ratio;
  out[idx] = in[(((b * kv_heads) + kh) * seq + s) * dim + d];
}
