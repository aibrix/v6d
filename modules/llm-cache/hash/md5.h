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

#ifndef MODULES_LLM_CACHE_HASH_MD5_H_
#define MODULES_LLM_CACHE_HASH_MD5_H_

#include <openssl/evp.h>
#include <cstdio>
#include <string>

namespace vineyard {

std::string md5(const std::string& content) {
  auto* context = EVP_MD_CTX_new();
  const auto* md = EVP_md5();
  unsigned char md_value[EVP_MAX_MD_SIZE];
  unsigned int md_len;
  std::string output;

  EVP_DigestInit_ex(context, md, nullptr);
  EVP_DigestUpdate(context, content.c_str(), content.size());
  EVP_DigestFinal_ex(context, md_value, &md_len);
  EVP_MD_CTX_free(context);

  output.resize(md_len * 2);
  for (int i = 0; i < md_len; i++) {
    std::sprintf(&output[i * 2], "%02x",  // NOLINT(runtime/printf)
                 md_value[i]);
  }
  return output;
}

}  // namespace vineyard

#endif  // MODULES_LLM_CACHE_HASH_MD5_H_
