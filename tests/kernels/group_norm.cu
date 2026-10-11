extern "C" __global__ void group_norm(const float *in, float *out, int channels,
                                      int size, int groups, float eps) {
  int group = blockIdx.x;
  if (group >= groups) {
    return;
  }
  int group_channels = channels / groups;
  float mean = 0.0F;
  float var = 0.0F;
  for (int c = group * group_channels; c < (group + 1) * group_channels; ++c) {
    for (int s = 0; s < size; ++s) {
      float v = in[c * size + s];
      mean += v;
      var += v * v;
    }
  }
  int count = group_channels * size;
  mean /= static_cast<float>(count);
  var = var / static_cast<float>(count) - mean * mean;
  for (int c = group * group_channels; c < (group + 1) * group_channels; ++c) {
    for (int s = 0; s < size; ++s) {
      out[c * size + s] = (in[c * size + s] - mean) / sqrtf(var + eps);
    }
  }
}
