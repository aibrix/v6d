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

#ifndef MODULES_LLM_CACHE_STORAGE_AIBRIX_BLOB_STORAGE_H_
#define MODULES_LLM_CACHE_STORAGE_AIBRIX_BLOB_STORAGE_H_

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "client/client.h"
#include "common/util/evicting_cache_map.h"
#include "common/util/logging.h"

#include "llm-cache/ds/kv_cache_chunk.h"
#include "llm-cache/hash/hasher.h"
#include "llm-cache/storage/storage.h"

namespace vineyard {

class AIBrixBlobStorage
    : public IStorage,
      public std::enable_shared_from_this<AIBrixBlobStorage> {
 private:
  static constexpr int kMaxTokensPerSeq = 64 * 1024;
  static constexpr double kSmallFifoCapacityRatio = 0.3;
  static constexpr int kMinEviction = 32;

  RPCClient& rpc_client_;
  Client& ipc_client_;

  std::shared_ptr<IHashAlgorithm> hash_alg_;
  std::shared_ptr<Hasher> hasher_;

  size_t tensor_nbytes_;
  int layer_;
  int chunk_size_;
  int capacity_;
  size_t chunk_obj_size_;
  std::string kv_cache_ns_;

  // intervals in seconds
  std::chrono::duration<int64_t> local_sync_interval_s_;
  std::chrono::duration<int64_t> global_gc_interval_s_;
  // TTL in seconds
  std::chrono::duration<int64_t> global_ttl_s_;

  bool exit_flag_ = false;

  // global GC is carried out in the global GC thread.
  // it checks global chunks' access time and deletes
  // the expired chunks.
  bool global_gc_enabled_ = false;
  std::condition_variable global_gc_cv_;
  std::mutex global_gc_mu_;
  std::thread global_gc_thread_;

  // local sync is carried out in the local sync thread.
  // it persists newly added chunks and deletes evicted chunks.
  const bool local_gc_enabled_ = true;
  std::condition_variable local_sync_cv_;
  std::mutex local_sync_mu_;
  std::thread local_sync_thread_;

  // S3-FIFO
  //
  // - a small FIFO map (S) that quickly removes new and unpopular
  //   objects (quick demotion)
  // - a main FIFO map (M) that keeps popular objects in the cache
  //   with reinsertion (lazy promotion), and
  // - a ghost FIFO map (G) that stores the id of objects recently
  //   evicted from S.
  //
  // G stores same number of entries as M. Note that, a request
  // found in G is not a cache hit since the items in G do not
  // have data.
  //
  // Each entry in the cache uses one bit of metadata to track
  // hotness.
  //
  // Upon a cache miss, if the id of the requested object is not
  // tracked in G, it is inserted into S; however, if the requested
  // object is tracked in G, then the object is inserted into M.
  //
  // When S performs an eviction if the object has not been reused
  // since insertion, it is evicted to G, and only its id is tracked
  // in G. Otherwise, the object is promoted to M with the access
  // bit reset to zero.
  //
  // When M performs an eviction, an object is directly evicted if
  // it has not been reused since the insertion. Otherwise, the
  // object is reinserted into M. (lazy promotion)
  struct FifoEntry {
    std::shared_ptr<KVCacheChunkBuilder> chunk_builder = nullptr;
    ObjectID object_id = InvalidObjectID();
    bool access_bit = false;
  };

  std::mutex ghost_fifo_mu_;
  EvictingCacheMap<std::string, FifoEntry> ghost_fifo_;
  std::mutex small_fifo_mu_;
  EvictingCacheMap<std::string, FifoEntry> small_fifo_;
  std::mutex main_fifo_mu_;
  EvictingCacheMap<std::string, FifoEntry>
      main_fifo_;  // mirror of global chunk list

  std::vector<ObjectID> evict_list_;

 public:
  AIBrixBlobStorage(RPCClient& rpc_client, Client& ipc_client,
                    size_t tensor_nbytes, int capacity, int layer,
                    int chunk_size, std::string kv_cache_ns,
                    int64_t local_sync_interval_s, bool global_gc_enabled,
                    int64_t global_gc_interval_s, int64_t global_ttl_s);

  Status Update(
      const std::vector<int>& token_list, int next_token,
      const std::vector<std::pair<LLMKV, LLMKV>>& kv_tensors) override {
    return Status::NotImplemented();
  }

  Status Update(
      const std::vector<int>& token_list,
      const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kv_tensors,
      size_t& updated) override;

  Status Update(
      const std::vector<int>& prefix, const std::vector<int>& token_list,
      const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kv_tensors,
      size_t& updated) override;

  Status Query(const std::vector<int>& token_list,
               std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kv_tensors,
               size_t& matched) override;

  Status Query(const std::vector<int>& prefix, int token,
               std::vector<std::pair<LLMKV, LLMKV>>& kv_tensors) override {
    return Status::NotImplemented();
  }

  Status Query(const std::vector<int>& prefix, const std::vector<int>& tokens,
               std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kv_tensors,
               size_t& matched) override;

  void CloseCache() override;

  ~AIBrixBlobStorage() = default;

  Status Init();

  void StopGlobalGCThread() override { global_gc_enabled_ = false; }

  void StartGlobalGCThread() override { global_gc_enabled_ = true; }

  static std::string GetTimestamp(
      std::chrono::duration<int64_t, std::nano> time);

  Status ListKVCache(const std::string& prefix, std::vector<ObjectMeta>& metas);

 private:
  template <class PairsT>
  void MainFifoPruneHookLoop(PairsT&& pairs,
                             std::vector<ObjectID>& delete_list);

  Status BuildMainFifo();

  ClientBase& GetClient();

  Status GetTokenChunkHashes(const std::vector<int>& prefix,
                             const std::vector<int>& tokens,
                             std::vector<std::string>& chunk_hashes);

  Status UpdateInternal(
      const std::vector<int>& prefix, const std::vector<int>& token_list,
      const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kv_tensors,
      size_t& updated);

  Status QueryInternal(
      const std::vector<int>& prefix, const std::vector<int>& tokens,
      std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kv_tensors,
      size_t& matched);

  Status SealAndPersist(
      const std::string& name,
      const std::shared_ptr<KVCacheChunkBuilder>& chunk_builder,
      ObjectID& chunk_id);

  Status Delete(const std::vector<std::string>& chunk_list);

  Status Delete(const std::vector<ObjectID>& obj_ids);

  Status LocalSyncFunc();

  Status GlobalGCFunc();

  Status ProcessPersistList(
      const std::vector<std::pair<std::string, FifoEntry>>& persist_list);

  Status ProcessUpdateList(
      const std::vector<std::pair<std::string, FifoEntry>>& udpate_list);

  static void LocalSyncThread(std::shared_ptr<AIBrixBlobStorage> self);

  static void GlobalGCThread(std::shared_ptr<AIBrixBlobStorage> self);

  void TerminateGCThreads();
};

}  // namespace vineyard

#endif  // MODULES_LLM_CACHE_STORAGE_AIBRIX_BLOB_STORAGE_H_
