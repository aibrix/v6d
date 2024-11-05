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

#ifndef MODULES_LLM_CACHE_DS_KV_CACHE_CHUNK_H_
#define MODULES_LLM_CACHE_DS_KV_CACHE_CHUNK_H_

#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <shared_mutex>
#include <string>
#include <utility>
#include <vector>

#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/i_object.h"
#include "client/ds/remote_blob.h"
#include "client/rpc_client.h"
#include "llm-cache/ds/kv_tensor.h"

namespace vineyard {

// forward declaration
struct LLMKV;

class KVCacheChunk : public vineyard::Registered<KVCacheChunk> {
 public:
  inline static constexpr char kFieldNameNS[] = "namespace";
  inline static constexpr char kFieldNameBuffer[] = "buffer";
  inline static constexpr char kFieldNameTotalTokens[] = "total_tokens";
  inline static constexpr char kFieldNameTensorNBytes[] = "tensor_nbytes";
  inline static constexpr char kFieldNameLayer[] = "layer";
  inline static constexpr char kFieldNameChunkSize[] = "chunk_size";
  inline static constexpr char kFieldNameAccessTime[] = "access_time";
  inline static constexpr char kFieldNameMd5[] = "md5";

 private:
  std::shared_ptr<Buffer> buffer_;
  int total_tokens_;
  int tensor_nbytes_;
  int layer_;
  int chunk_size_;
  uint64_t access_time_;
  std::string md5_;
  std::string ns_;

 public:
  static std::unique_ptr<Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<Object>(
        std::unique_ptr<KVCacheChunk>{new KVCacheChunk()});
  }

  void Construct(const ObjectMeta& meta) override;

  int GetChunkSize() { return chunk_size_; }

  static std::string GetNameSpace(const std::string& kv_cache_ns) {
    return std::regex_replace(kv_cache_ns, std::regex("_+$"), "");
  }

  ~KVCacheChunk() = default;

  friend class KVCacheChunkBuilder;
};

class KVCacheChunkBuilder {
 private:
  RPCClient& rpc_client_;
  std::vector<int> all_tokens_;
  std::shared_ptr<RemoteBlobWriter> remote_buffer_writer_ = nullptr;
  ObjectID chunk_id_;
  std::shared_ptr<Buffer> buffer_ = nullptr;
  int total_tokens_;
  int tensor_nbytes_;
  int layer_;
  int chunk_size_;
  std::string ns_;
  std::shared_mutex time_mu_;
  uint64_t g_access_time_ = 0;
  uint64_t access_time_ = 0;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::atomic<bool> is_ready_ = false;
  std::string md5_;

 public:
  static Status Make(std::shared_ptr<KVCacheChunkBuilder>& chunk_builder,
                     RPCClient& rpc_client, int max_tokens, int tensor_nbytes,
                     int layer, int chunk_size, const std::string& kv_cache_ns);

  static Status Make(std::shared_ptr<KVCacheChunkBuilder>& chunk_builder,
                     RPCClient& rpc_client, int tensor_nbytes, int layer,
                     int chunk_size, const std::string& kv_cache_ns,
                     ObjectID chunk_id);

  Status Update(
      const std::vector<int>& prefix, const std::vector<int>& tokens,
      const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kv_tensor);

  Status Query(const std::vector<int>& prefix, const std::vector<int>& tokens,
               std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kv_tensor);

  void SetAccessTime(uint64_t time) {
    std::unique_lock<std::shared_mutex> wlock(time_mu_);
    if (time > access_time_) {
      access_time_ = time;
    }
  }

  void SetGlobalAccessTime(uint64_t time) {
    std::unique_lock<std::shared_mutex> wlock(time_mu_);
    if (time > g_access_time_) {
      g_access_time_ = time;
    }
  }

  uint64_t GetGlobalAccessTime() {
    std::shared_lock<std::shared_mutex> rlock(time_mu_);
    return g_access_time_;
  }

  uint64_t GetAccessTime() {
    std::shared_lock<std::shared_mutex> rlock(time_mu_);
    return access_time_;
  }

  bool IsReady() { return is_ready_; }

  std::shared_ptr<Object> Seal();

  uint64_t GetTensorNBytes() { return tensor_nbytes_; }

  int GetChunkSize() { return chunk_size_; }

  void PrintKVCacheChunk();

  Status IsSame(const ObjectMeta& meta);

  KVCacheChunkBuilder(RPCClient& rpc_client, int tensor_nbytes, int layer,
                      int chunk_size, const std::string& kv_cache_ns)
      : rpc_client_(rpc_client),
        tensor_nbytes_(tensor_nbytes),
        layer_(layer),
        chunk_size_(chunk_size),
        ns_(KVCacheChunk::GetNameSpace(kv_cache_ns)) {}

  ~KVCacheChunkBuilder() = default;

 private:
  Status Construct();

  Status UpdateImpl(
      const std::vector<int>& prefix, const std::vector<int>& tokens,
      const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kv_tensor);

  Status QueryImpl(
      const std::vector<int>& prefix, const std::vector<int>& tokens,
      std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kv_tensor);

  Status QueryInternal(
      const std::vector<int>& prefix, const std::vector<int>& tokens,
      std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kv_tensor);
};

}  // namespace vineyard

#endif  // MODULES_LLM_CACHE_DS_KV_CACHE_CHUNK_H_
