#include "mem/atomics.h"

#include <atomic>
#include <bit>

namespace cudasim::mem {

void atomic_add_u32(std::uint32_t *address, std::uint32_t value) {
  auto *atomic_address =
      reinterpret_cast<std::atomic<std::uint32_t> *>(address);
  atomic_address->fetch_add(value, std::memory_order_relaxed);
}

void atomic_add_f32(float *address, float value) {
  auto *atomic_address =
      reinterpret_cast<std::atomic<std::uint32_t> *>(address);
  for (;;) {
    std::uint32_t old_bits = atomic_address->load(std::memory_order_relaxed);
    const float old_value = std::bit_cast<float>(old_bits);
    const std::uint32_t new_bits =
        std::bit_cast<std::uint32_t>(old_value + value);
    if (atomic_address->compare_exchange_weak(old_bits, new_bits,
                                              std::memory_order_relaxed)) {
      return;
    }
  }
}

} // namespace cudasim::mem
