extern "C" __global__ void embedding_lookup(const float *weight,
                                            const int *indices,
                                            float *output, int count,
                                            int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  int token = indices[i];
  for (int d = 0; d < dim; ++d) {
    output[i * dim + d] = weight[token * dim + d];
  }
}
