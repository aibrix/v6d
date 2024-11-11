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

// A KVCacheChunk contains all the KV tensors of a fixed number of
// tokens (i.e., `chunk_size`).
//
// In its object blob, we first store all the KV tensors, and then
// store all the tokens (including prefix tokens and current tokens
// cached in the chunk), which will be used to avoid hash conflicts.
//
// In its metadata, we store the namespace (i.e., `ns_`), which will
// be used as the name prefix of each chunk. Clients can also use the
// namespace to list all the chunks. Access time (i.e., 'access_time_`)
// in its metadata is used for the TTL-based global GC. We also have
// the md5sum of all tokens (including prefix tokens and current tokens)
// in its metadata. When we reconstruct a chunk from the object blob
// and metadata, we calculate the md5sum of all tokens in the blob and
// compare it with the md5sum in the metadata. If they are the same,
// we consider the chunk is valid. Otherwise, we consider the chunk is
// corrupted. By far, we don't use the md5sum of the tensors to alleviate
// the compute overhead.
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
  // number of prefix tokens and current tokens in the chunk
  int total_tokens_;
  int tensor_nbytes_;
  int layer_;
  int chunk_size_;
  // access time is used for TTL-based global GC
  uint64_t access_time_;
  // md5sum of all tokens (including prefix tokens and current tokens)
  std::string md5_;
  // namespace. chunks within the same namespace will be shared
  // among different clients
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

// A KVCacheChunkBuilder is used to build a KVCacheChunk.
//
// We have two kinds of builders:
// 1. The builder to build a new chunk.
// 2. The builder to rebuild a chunk from the object blob and metadata.
//
// For the first kind of builder, `Make` creates an empty chunk and an
// `Update` filles the chunk with KV tensors. After `Update`, the chunk
// is marked as ready and waiting readers will be notified. This kind
// of builder can be sealed to a KVCacheChunk.
//
// For the second kind of builder, `Make` only assignes the chunk id and
// the first `Query` will trigger a construction of the chunk, i.e.,
// constructing the corresponding chunk with fetched metadata and blob.
// After construction, the chunk is marked as ready and other waiting
// readers will be notified. This kind of builder will never be sealed
// since the chunk already exists in the object store.
//
// We also track the access time of the chunk in the builder. Global
// access time is the latest access time of the global object we know.
// Access time is the local access time that is updated by each access.
// The local access time will finally be updated to the global access
// time based on the policy used in AIBrixBlobStorage.
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

  // `time_mu_` protects the access times of the chunk.
  std::shared_mutex time_mu_;
  uint64_t g_access_time_ = 0;
  uint64_t access_time_ = 0;

  // `mutex_` and `cv_` are used to block readers until the chunk
  // is ready to be read.
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

  // Whether the chunk is ready to be read.
  bool IsReady() { return is_ready_; }

  std::shared_ptr<Object> Seal();

  uint64_t GetTensorNBytes() { return tensor_nbytes_; }

  int GetChunkSize() { return chunk_size_; }

  void PrintKVCacheChunk();

  // Whether the chunk is the same as the chunk with the given metadata.
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
