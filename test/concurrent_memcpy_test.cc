/** Copyright 2020-2023 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <cstring>
#include <memory>
#include <string>
#include <thread>

#include "common/memory/memcpy.h"
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

void testing_memcpy(const size_t size) {
  LOG(INFO) << "Testing memcpy with size: " << size;
  std::unique_ptr<char[]> src(new char[size]);
  std::unique_ptr<char[]> dst(new char[size]);
  for (size_t i = 0; i < size; ++i) {
    src[i] = i % 256;
  }

  std::vector<size_t> sizes_to_test = {
      size, size - 1, size - 3, size - 10, size - 1024, size - 1024 * 1024,
  };

  for (size_t size_to_test : sizes_to_test) {
    if (size_to_test > size) {
      continue;
    }
    for (size_t concurrency = 1; concurrency <= 8; ++concurrency) {
      memset(dst.get(), 0, size_to_test);
      memory::concurrent_memcpy(dst.get(), src.get(), size_to_test,
                                concurrency);
      CHECK_EQ(0, memcmp(dst.get(), src.get(), size_to_test));
    }
  }
  LOG(INFO) << "Passed memcpy test with size: " << size;
}

void test_concurrent_memcpy_n_basic() {
    std::vector<const void*> src_buffers;
    std::vector<void*> dst_buffers;
    size_t buffer_size = 1 * 1024 * 1024;  // 1MB per buffer

    char* src1 = new char[buffer_size];
    char* dst1 = new char[buffer_size];

    std::memset(src1, 'A', buffer_size);

    src_buffers.push_back(src1);
    dst_buffers.push_back(dst1);

    size_t concurrency_level = 1;

    vineyard::memory::concurrent_memcpy_n(dst_buffers, src_buffers, buffer_size,
                                          concurrency_level);

    assert(std::memcmp(dst1, src1, buffer_size) == 0);

    delete[] src1;
    delete[] dst1;

    LOG(INFO) << "Passed concurrent_memcpy_n test: basic single buffer copy";
}

void test_concurrent_memcpy_n_multiple_buffers() {
    std::vector<const void*> src_buffers;
    std::vector<void*> dst_buffers;
    size_t buffer_size = 2 * 1024 * 1024;  // 2MB per buffer

    char* src1 = new char[buffer_size];
    char* src2 = new char[buffer_size];
    char* dst1 = new char[buffer_size];
    char* dst2 = new char[buffer_size];

    std::memset(src1, 'A', buffer_size);
    std::memset(src2, 'B', buffer_size);

    src_buffers.push_back(src1);
    src_buffers.push_back(src2);
    dst_buffers.push_back(dst1);
    dst_buffers.push_back(dst2);

    size_t concurrency_level = 2;

    vineyard::memory::concurrent_memcpy_n(dst_buffers, src_buffers, buffer_size,
                                          concurrency_level);

    assert(std::memcmp(dst1, src1, buffer_size) == 0);
    assert(std::memcmp(dst2, src2, buffer_size) == 0);

    delete[] src1;
    delete[] src2;
    delete[] dst1;
    delete[] dst2;

    LOG(INFO) << "Passed concurrent_memcpy_n test: multiple buffer copy";
}

void test_concurrent_memcpy_n_with_small_buffers() {
    std::vector<const void*> src_buffers;
    std::vector<void*> dst_buffers;
    size_t buffer_size = 1 * 1024 * 1024;  // 1MB per buffer

    char* src1 = new char[buffer_size];
    char* src2 = new char[buffer_size];
    char* dst1 = new char[buffer_size];
    char* dst2 = new char[buffer_size];

    std::memset(src1, 'A', buffer_size);
    std::memset(src2, 'B', buffer_size);

    src_buffers.push_back(src1);
    src_buffers.push_back(src2);
    dst_buffers.push_back(dst1);
    dst_buffers.push_back(dst2);

    size_t concurrency_level = 2;

    vineyard::memory::concurrent_memcpy_n(dst_buffers, src_buffers, buffer_size,
                                          concurrency_level);

    assert(std::memcmp(dst1, src1, buffer_size) == 0);
    assert(std::memcmp(dst2, src2, buffer_size) == 0);

    delete[] src1;
    delete[] src2;
    delete[] dst1;
    delete[] dst2;

    LOG(INFO) << "Passed concurrent_memcpy_n test: small buffers";
}

void test_concurrent_memcpy_n_large_buffer() {
    std::vector<const void*> src_buffers;
    std::vector<void*> dst_buffers;
    size_t buffer_size = 8 * 1024 * 1024;  // 8MB per buffer

    char* src1 = new char[buffer_size];
    char* dst1 = new char[buffer_size];

    std::memset(src1, 'A', buffer_size);

    src_buffers.push_back(src1);
    dst_buffers.push_back(dst1);

    size_t concurrency_level = 2;  // Expect splitting across threads

    vineyard::memory::concurrent_memcpy_n(dst_buffers, src_buffers, buffer_size,
                                          concurrency_level);

    assert(std::memcmp(dst1, src1, buffer_size) == 0);

    delete[] src1;
    delete[] dst1;

    LOG(INFO)
        << "Passed concurrent_memcpy_n test: large buffer copy with splitting";
}

void test_concurrent_memcpy_n_with_uneven_distribution() {
    std::vector<const void*> src_buffers;
    std::vector<void*> dst_buffers;
    size_t buffer_size = 3 * 1024 * 1024;  // 3MB per buffer

    char* src1 = new char[buffer_size];
    char* src2 = new char[buffer_size];
    char* src3 = new char[buffer_size];
    char* dst1 = new char[buffer_size];
    char* dst2 = new char[buffer_size];
    char* dst3 = new char[buffer_size];

    std::memset(src1, 'A', buffer_size);
    std::memset(src2, 'B', buffer_size);
    std::memset(src3, 'C', buffer_size);

    src_buffers.push_back(src1);
    src_buffers.push_back(src2);
    src_buffers.push_back(src3);
    dst_buffers.push_back(dst1);
    dst_buffers.push_back(dst2);
    dst_buffers.push_back(dst3);

    size_t concurrency_level = 2;  // Results in uneven distribution across threads

    vineyard::memory::concurrent_memcpy_n(dst_buffers, src_buffers, buffer_size,
                                          concurrency_level);

    assert(std::memcmp(dst1, src1, buffer_size) == 0);
    assert(std::memcmp(dst2, src2, buffer_size) == 0);
    assert(std::memcmp(dst3, src3, buffer_size) == 0);

    delete[] src1;
    delete[] src2;
    delete[] src3;
    delete[] dst1;
    delete[] dst2;
    delete[] dst3;

    LOG(INFO) << "Passed concurrent_memcpy_n test: uneven distribution with "
                 "larger buffers";
}

int main(int argc, char** argv) {
  if (argc < 1) {
    printf("usage ./concurrent_memcpy_test");
    return 1;
  }

  for (size_t sz = 1024 * 1024 * 8; sz < 1024 * 1024 * 1024;
       sz += 1024 * 1024 * 256) {
    testing_memcpy(sz);
  }

  test_concurrent_memcpy_n_basic();
  test_concurrent_memcpy_n_multiple_buffers();
  test_concurrent_memcpy_n_with_small_buffers();
  test_concurrent_memcpy_n_large_buffer();
  test_concurrent_memcpy_n_with_uneven_distribution();

  LOG(INFO) << "Passed concurrent memcpy tests...";

  return 0;
}
