extern "C" __global__ void concat(const float *a, const float *b, float *out,
                                  int a_size, int b_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < a_size) {
    out[i] = a[i];
  } else if (i < a_size + b_size) {
    out[i] = b[i - a_size];
  }
}
