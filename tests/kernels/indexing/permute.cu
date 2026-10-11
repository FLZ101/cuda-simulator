extern "C" __global__ void permute(
    const float *a, const float *b, const float *c, const float *d,
    const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
    int p5, int p6, int p7, float f0, float f1, float f2, float f3) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int d0 = p1;
  const int d1 = p2;
  const int d2 = p3;
  const int out_d0 = p4 == 0 ? d0 : (p4 == 1 ? d1 : d2);
  const int out_d1 = p5 == 0 ? d0 : (p5 == 1 ? d1 : d2);
  const int out_d2 = p6 == 0 ? d0 : (p6 == 1 ? d1 : d2);
  for (int o0 = 0; o0 < out_d0; ++o0) {
    for (int o1 = 0; o1 < out_d1; ++o1) {
      for (int o2 = 0; o2 < out_d2; ++o2) {
        int i0 = 0;
        int i1 = 0;
        int i2 = 0;
        if (p4 == 0) {
          i0 = o0;
        } else if (p4 == 1) {
          i1 = o0;
        } else {
          i2 = o0;
        }
        if (p5 == 0) {
          i0 = o1;
        } else if (p5 == 1) {
          i1 = o1;
        } else {
          i2 = o1;
        }
        if (p6 == 0) {
          i0 = o2;
        } else if (p6 == 1) {
          i1 = o2;
        } else {
          i2 = o2;
        }
        const int out_idx = (o0 * out_d1 + o1) * out_d2 + o2;
        out[out_idx] = a[(i0 * d1 + i1) * d2 + i2];
      }
    }
  }
}
