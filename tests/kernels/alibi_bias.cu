extern "C" __global__ void alibi_bias(const float *in, float *out, int heads,
                                      int seq, int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = heads * seq * dim;
  if (i < total) {
    int head = i / (seq * dim);
    int pos = (i / dim) % seq;
    float slope = static_cast<float>(head + 1) * -0.1F;
    out[i] = in[i] + slope * static_cast<float>(pos);
  }
}
