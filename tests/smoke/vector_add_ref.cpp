#include <cstddef>
#include <vector>

static std::vector<float> add_vectors(const std::vector<float> &a,
                                      const std::vector<float> &b) {
  std::vector<float> c(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    c[i] = a[i] + b[i];
  }
  return c;
}

int main() {
  const std::vector<float> a{1.0F, 2.0F, 3.0F};
  const std::vector<float> b{4.0F, 5.0F, 6.0F};
  const std::vector<float> c = add_vectors(a, b);

  const std::vector<float> expected{5.0F, 7.0F, 9.0F};
  if (c != expected) {
    return 1;
  }

  return 0;
}
