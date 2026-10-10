#include "core/version.h"

#include <iostream>

int main() {
  std::cout << cudasim::version_string() << '\n';
  return 0;
}
