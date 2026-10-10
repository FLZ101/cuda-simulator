#pragma once

#include "mem/access_pattern.h"
#include "mem/device_memory.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cudasim::mem {

AccessPattern classify(const std::vector<std::size_t> &offsets);

AccessPattern classify_pointers(const std::vector<const void *> &addresses);

void coalesced_read_u32(const DeviceBuffer &buffer,
                        const std::vector<std::size_t> &offsets,
                        std::vector<std::uint32_t> &output);

void coalesced_write_u32(DeviceBuffer &buffer,
                         const std::vector<std::size_t> &offsets,
                         const std::vector<std::uint32_t> &values);

void coalesced_read_u32_pointers(const std::vector<const void *> &addresses,
                                 std::vector<std::uint32_t> &output);

void coalesced_write_u32_pointers(const std::vector<void *> &addresses,
                                  const std::vector<std::uint32_t> &values);

} // namespace cudasim::mem
