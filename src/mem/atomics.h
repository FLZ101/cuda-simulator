#pragma once

#include <cstdint>

namespace cudasim::mem {

void atomic_add_u32(std::uint32_t *address, std::uint32_t value);
void atomic_add_f32(float *address, float value);

} // namespace cudasim::mem
