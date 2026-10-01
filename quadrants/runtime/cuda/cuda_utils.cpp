#include "quadrants/runtime/cuda/cuda_utils.h"
#include "quadrants/rhi/cuda/cuda_context.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <tuple>

namespace quadrants::lang {
namespace cuda {

bool on_cuda_device(void *ptr) {
  unsigned int attr_val = 0;
  uint32_t ret_code =
      CUDADriver::get_instance().mem_get_attribute.call(&attr_val, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (void *)ptr);
  return ret_code == CUDA_SUCCESS && attr_val == CU_MEMORYTYPE_DEVICE;
}

int occupancy_grid_dim(void *kernel, int requested_grid_dim, int block_dim, std::size_t dynamic_shared_array_bytes) {
  if (kernel == nullptr || requested_grid_dim <= 1 || block_dim <= 0) {
    return std::max(requested_grid_dim, 1);
  }

  using CacheKey = std::tuple<void *, int, std::size_t>;
  static std::mutex cache_mutex;
  static std::map<CacheKey, int> resident_grid_cache;

  const CacheKey key{kernel, block_dim, dynamic_shared_array_bytes};
  int resident_grid_dim = 0;
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto cached = resident_grid_cache.find(key);
    if (cached != resident_grid_cache.end()) {
      resident_grid_dim = cached->second;
    }
  }

  if (resident_grid_dim == 0) {
    auto &driver = CUDADriver::get_instance();
    if (dynamic_shared_array_bytes > 0) {
      driver.kernel_set_attribute(kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                  static_cast<int>(dynamic_shared_array_bytes));
    }

    int active_blocks_per_sm = 0;
    driver.kernel_get_occupancy(&active_blocks_per_sm, kernel, block_dim, dynamic_shared_array_bytes);
    int num_sms = 1;
    driver.device_get_attribute(&num_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, nullptr);
    resident_grid_dim = std::max(active_blocks_per_sm * num_sms, 1);

    std::lock_guard<std::mutex> lock(cache_mutex);
    resident_grid_cache.emplace(key, resident_grid_dim);
  }

  return std::min(requested_grid_dim, resident_grid_dim);
}

}  // namespace cuda
}  // namespace quadrants::lang
