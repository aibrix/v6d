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
#include "common/memory/cuda.h"

#include "common/util/env.h"
#include "common/util/likely.h"
#include "common/util/logging.h"

#if defined(ENABLE_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#endif

int v6d_cuda_host_register(const void* ptr, size_t size, bool readonly) {
#if defined(ENABLE_CUDA)
  cudaError_t rc = cudaSuccess;
  if ((rc = cudaSetDeviceFlags(cudaDeviceMapHost)) != cudaSuccess) {
    LOG(ERROR) << "cudaDeviceMapHost failed: " << cudaGetErrorString(rc);
    return rc;
  }

  int flags = readonly ? cudaHostRegisterReadOnly : cudaHostRegisterMapped;
  rc = cudaHostRegister(const_cast<void*>(ptr), size, flags);
  if (rc != cudaSuccess) {
    LOG(ERROR) << "cudaHostRegister failed: " << cudaGetErrorString(rc);
    return rc;
  }

  // If host and device pointer don't match, return an error
  void* devptr = nullptr;
  rc = cudaHostGetDevicePointer(&devptr, const_cast<void*>(ptr), 0);
  if (rc != cudaSuccess) {
    v6d_cuda_host_unregister(ptr);
    LOG(ERROR) << "cudaHostGetDevicePointer failed: " << cudaGetErrorString(rc);
    return rc;
  }

  if (devptr != const_cast<void*>(ptr)) {
    v6d_cuda_host_unregister(ptr);
    LOG(ERROR) << "Host and device pointer don't match. Please don't use CUDA "
                  "managed memory.";
    return -1;
  }

  VLOG(100) << "Registered " << vineyard::prettyprint_memory_size(size)
            << " of CUDA managed memory.";

  return rc;
#else
  return -1;
#endif
}

int v6d_cuda_host_unregister(const void* ptr) {
#if defined(ENABLE_CUDA)
  return cudaHostUnregister(const_cast<void*>(ptr));
#else
  return -1;
#endif
}
