#include "mem/atomics.h"
#include "mem/coalescer.h"
#include "mem/device_memory.h"

#include <cstdint>
#include <vector>

int main() {
  cudasim::mem::DeviceBuffer buffer(128);

  std::vector<std::size_t> contiguous;
  for (std::size_t i = 0; i < 8; ++i) {
    contiguous.push_back(i * sizeof(std::uint32_t));
  }
  if (cudasim::mem::classify(contiguous) !=
      cudasim::mem::AccessPattern::kContiguous) {
    return 1;
  }

  std::vector<std::uint32_t> values{1, 2, 3, 4, 5, 6, 7, 8};
  cudasim::mem::coalesced_write_u32(buffer, contiguous, values);
  std::vector<std::uint32_t> output;
  cudasim::mem::coalesced_read_u32(buffer, contiguous, output);
  if (output != values) {
    return 2;
  }

  std::vector<std::size_t> strided{0, 16, 32, 48};
  if (cudasim::mem::classify(strided) !=
      cudasim::mem::AccessPattern::kStrided) {
    return 3;
  }

  std::uint32_t atomic_value = 0;
  cudasim::mem::atomic_add_u32(&atomic_value, 5);
  cudasim::mem::atomic_add_u32(&atomic_value, 7);
  if (atomic_value != 12) {
    return 4;
  }

  return 0;
}
