/** Copyright 2024 AIBrix.

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
#ifndef SRC_COMMON_MEMORY_CUDA_H_
#define SRC_COMMON_MEMORY_CUDA_H_

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Make the memory pointing by ptr managed by CUDA.
//
// @param ptr: the memory pointer.
// @param size: the size of the memory region
// @param readonly: register as readonly region
//
// @return 0 on success, other values (cudaError_t) otherwise.
int v6d_cuda_host_register(const void* ptr, size_t size, bool readonly = false);

// Unregister the memory pointed by ptr from CUDA.
//
// @param ptr: the memory pointer.
//
// @return 0 on success, other values (cudaError_t) otherwise.
int v6d_cuda_host_unregister(const void* ptr);

#ifdef __cplusplus
}
#endif

#endif  // SRC_COMMON_MEMORY_CUDA_H_
