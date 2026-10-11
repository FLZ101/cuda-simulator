extern "C" __global__ void threshold(const float *in, float *out, int n,
                                     float threshold_value, float value) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx] > threshold_value ? in[idx] : value;
  }
}
