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

#include "common/util/logging.h"

// Google glog's api does not have an external function that allows one to check
// if glog is initialized or not. It does have an internal function - so we are
// declaring it here. This is a hack but has been used by a bunch of others too
// (e.g. Torch).
namespace google {
namespace glog_internal_namespace_ {
bool IsGoogleLoggingInitialized();
}  // namespace glog_internal_namespace_

bool IsLoggingInitialized() {
  return google::glog_internal_namespace_::IsGoogleLoggingInitialized();
}
}  // namespace google
