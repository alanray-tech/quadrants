#pragma once

#include <cstddef>

namespace quadrants::lang {
namespace cuda {

bool on_cuda_device(void *ptr);

// Clamp a grid-stride kernel to one resident wave using the compiled
// function's actual register/shared-memory occupancy. The result never
// exceeds requested_grid_dim, preserving explicit smaller launch policies.
int occupancy_grid_dim(void *kernel, int requested_grid_dim, int block_dim, std::size_t dynamic_shared_array_bytes);

}  // namespace cuda
}  // namespace quadrants::lang
