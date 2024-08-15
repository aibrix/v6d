/**
 * NOLINT(legal/copyright)
 *
 * The file src/common/memory/memcpy.h is referred and derived from project
 * clickhouse,
 *
 *    https://github.com/ClickHouse/ClickHouse/blob/master/base/glibc-compatibility/memcpy/memcpy.h
 *
 * which has the following license:
 *
 * Copyright 2016-2022 ClickHouse, Inc.
 *
 *             Apache License
 *      Version 2.0, January 2004
 * http://www.apache.org/licenses/
 */

#ifndef SRC_COMMON_MEMORY_MEMCPY_H_
#define SRC_COMMON_MEMORY_MEMCPY_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <vector>

namespace vineyard {

namespace memory {

static inline void * inline_memcpy(void * __restrict dst_, const void * __restrict src_, size_t size) {
    return memcpy(dst_, src_, size);
}

// clang-format on

// use the same default concurrency as apache-arrow.
static constexpr size_t default_memcpy_concurrency = 6;

static inline void* concurrent_memcpy(void* __restrict dst_,
                                      const void* __restrict src_, size_t size,
                                      const size_t concurrency = default_memcpy_concurrency) {
  static constexpr size_t concurrent_memcpy_threshold = 1024 * 1024 * 4;
  if (size < concurrent_memcpy_threshold) {
    inline_memcpy(dst_, src_, size);
  } else if ((dst_ >= src_ &&
              dst_ <= static_cast<const uint8_t*>(src_) + size) ||
             (src_ >= dst_ && src_ <= static_cast<uint8_t*>(dst_) + size)) {
    inline_memcpy(dst_, src_, size);
  } else {
    static constexpr size_t alignment = 1024 * 1024 * 4;
    size_t chunk_size = (size / concurrency + alignment - 1) & ~(alignment - 1);
    std::vector<std::thread> threads;
    for (size_t i = 0; i < concurrency; ++i) {
      if (size <= i * chunk_size) {
        break;
      }
      size_t chunk = std::min(chunk_size, size - i * chunk_size);
      threads.emplace_back([=]() {
        inline_memcpy(static_cast<uint8_t*>(dst_) + i * chunk_size,
                      static_cast<const uint8_t*>(src_) + i * chunk_size,
                      chunk);
      });
    }
    for (auto &thread: threads) {
      thread.join();
    }
  }
  return dst_;
}

static inline void concurrent_memcpy_n(
    const std::vector<void*>& dst_buffers,
    const std::vector<const void*>& src_buffers,
    size_t size_of_each_buffer,
    const size_t concurrency = default_memcpy_concurrency) {
  assert(dst_buffers.size() == src_buffers.size());
  
  size_t nbuffers = dst_buffers.size();
  static constexpr size_t concurrent_memcpy_threshold = 1024 * 1024 * 4;
  size_t total_size = size_of_each_buffer * nbuffers;

  if (total_size < concurrent_memcpy_threshold) {
    for (size_t i = 0; i < dst_buffers.size(); i++) {
      inline_memcpy(dst_buffers[i], src_buffers[i], size_of_each_buffer);
    }
  } else {
    static constexpr size_t alignment = 1024 * 1024 * 4;
    size_t chunk_size =
        (total_size / concurrency + alignment - 1) & ~(alignment - 1);

    size_t curr_start = 0;
    std::vector<std::thread> threads;

    for (size_t i = 0; i < concurrency; ++i) {
      size_t curr_end = std::min(curr_start + chunk_size, total_size);
      size_t start_idx = curr_start / size_of_each_buffer;
      size_t start_offset = curr_start % size_of_each_buffer;
      size_t end_idx = curr_end / size_of_each_buffer;
      size_t end_offset = curr_end % size_of_each_buffer;

      threads.emplace_back([=]() {
        for (size_t buff_idx = start_idx; buff_idx <= end_idx; ++buff_idx) {
          size_t start = (buff_idx == start_idx) ? start_offset : 0;
          size_t end = (buff_idx == end_idx) ? end_offset : size_of_each_buffer;
          inline_memcpy(
              static_cast<uint8_t*>(dst_buffers[buff_idx]) + start,
              static_cast<const uint8_t*>(src_buffers[buff_idx]) + start,
              end - start);
        }
      });

      curr_start = curr_end;
      if (curr_start >= total_size) {
        break;
      }
    }

    for (auto &thread: threads) {
      thread.join();
    }
  }
}

}  // namespace memory

}  // namespace vineyard

#endif  // SRC_COMMON_MEMORY_MEMCPY_H_
