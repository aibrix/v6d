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

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "llm-cache/storage/aibrix_blob_storage.h"
#include "llm-cache/thread_group.h"

namespace vineyard {
AIBrixBlobStorage::AIBrixBlobStorage(
    RPCClient& rpc_client, Client& ipc_client, size_t tensor_nbytes,
    int capacity, int layer, int chunk_size, std::string kv_cache_ns,
    int64_t local_sync_interval_s, bool global_gc_enabled,
    int64_t global_gc_interval_s, int64_t global_ttl_s)
    : rpc_client_(rpc_client),
      ipc_client_(ipc_client),
      hash_alg_(std::make_shared<MurmurHash3Algorithm>()),
      hasher_(std::make_shared<Hasher>(hash_alg_.get())),
      tensor_nbytes_(tensor_nbytes),
      capacity_(capacity),
      layer_(layer),
      chunk_size_(chunk_size),
      chunk_obj_size_(tensor_nbytes * 2 * layer * chunk_size +
                      kMaxTokensPerSeq * sizeof(int)),
      kv_cache_ns_(kv_cache_ns),
      local_sync_interval_s_(std::chrono::seconds(local_sync_interval_s)),
      global_gc_enabled_(global_gc_enabled),
      global_gc_interval_s_(std::chrono::seconds(global_gc_interval_s)),
      global_ttl_s_(std::chrono::seconds(global_ttl_s)),
      ghost_fifo_(capacity_),
      small_fifo_(capacity_ * kSmallFifoCapacityRatio),
      main_fifo_(capacity_ - capacity_ * kSmallFifoCapacityRatio,
                 kMinEviction) {
  kv_cache_ns_ = std::regex_replace(kv_cache_ns_, std::regex("/"), "_");
  kv_cache_ns_ = std::regex_replace(kv_cache_ns_ + "_", std::regex("_+"), "_");
}

Status AIBrixBlobStorage::Init() {
  if (!rpc_client_.Connected()) {
    RETURN_ON_ASSERT(ipc_client_.Connected());
    // check if rpc is enabled on server side
    std::map<InstanceID, json> cluster_info;
    RETURN_ON_ERROR(ipc_client_.ClusterInfo(cluster_info));
    auto instance_id = ipc_client_.instance_id();
    std::string rpc_endpoint =
        cluster_info[instance_id].value("rpc_endpoint", "");
    RETURN_ON_ASSERT(!rpc_endpoint.empty());
    std::string rdma_endpoint =
        cluster_info[instance_id].value("rdma_endpoint", "");
    RETURN_ON_ERROR(rpc_client_.Connect(rpc_endpoint, "", "", rdma_endpoint));
  }

  RETURN_ON_ASSERT(rpc_client_.Connected());

  main_fifo_.setPruneHook([this](auto&& pairs) {
    std::vector<ObjectID> delete_list;
    MainFifoPruneHookLoop(std::forward<decltype(pairs)>(pairs), delete_list);

    auto status = this->Delete(delete_list);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to delete objects, " << status.ToString();
    }
  });

  small_fifo_.setPruneHook([this](auto&& pairs) {
    for (auto& pair : pairs) {
      auto& key = pair.first;
      auto& entry = pair.second;
      VLOG(100) << "Evicting " << key << " from S";
      // Items in S should not have been persisted
      VINEYARD_ASSERT(entry.object_id == InvalidObjectID());
      if (entry.access_bit) {
        VLOG(100) << key << " has been accessed, promote it to M";
        // Promote the item to M
        entry.access_bit = false;

        std::unique_lock<std::mutex> lock(this->main_fifo_mu_);
        this->main_fifo_.set(key, std::move(entry), /* promote */ false);
      } else {
        VLOG(100) << "Evict " << key << " to G";
        // Evict the item to G
        entry.chunk_builder.reset();
        entry.access_bit = false;

        std::unique_lock<std::mutex> lock(this->ghost_fifo_mu_);
        this->ghost_fifo_.set(key, std::move(entry), /* promote */ false);
      }
    }
  });

  ghost_fifo_.setPruneHook([this](auto&& pairs) {
    for (auto& pair : pairs) {
      auto& key = pair.first;
      auto& entry = pair.second;
      VLOG(100) << "Evicting " << key << " from G";
      // Items in G should not have been persisted
      VINEYARD_ASSERT(entry.object_id == InvalidObjectID());
      entry.chunk_builder.reset();
    }
  });

  VINEYARD_DISCARD(BuildMainFifo());

  global_gc_thread_ =
      std::thread(AIBrixBlobStorage::GlobalGCThread, shared_from_this());
  local_sync_thread_ =
      std::thread(AIBrixBlobStorage::LocalSyncThread, shared_from_this());

  return Status::OK();
}

template <class PairsT>
void AIBrixBlobStorage::MainFifoPruneHookLoop(
    PairsT&& pairs, std::vector<ObjectID>& delete_list) {
  for (auto& pair : pairs) {
    auto& key = pair.first;
    auto& entry = pair.second;

    VLOG(100) << "Evicting " << key << " from M";

    if (entry.access_bit) {
      VLOG(100) << key << " has been accessed, reinsert it back to M";
      entry.access_bit = false;
      this->main_fifo_.set(key, std::move(entry));
    } else {
      // If the entry has a valid object id, then it has been persisted to the
      // object store, we have to explicitly delete it.
      if (entry.object_id != InvalidObjectID()) {
        VLOG(100) << "Deleting obj=" << entry.object_id;
        delete_list.push_back(entry.object_id);
      }

      entry.chunk_builder.reset();
    }
  }
}

Status AIBrixBlobStorage::BuildMainFifo() {
  std::vector<ObjectMeta> chunk_metas;
  RETURN_ON_ERROR(ListKVCache(kv_cache_ns_, chunk_metas));
  VLOG(100) << "Global M: " << chunk_metas.size() << " chunks";

  std::vector<ObjectID> delete_list;
  auto build_prune_hook = [this, &delete_list](auto&& pairs) {
    MainFifoPruneHookLoop(std::forward<decltype(pairs)>(pairs), delete_list);
  };

  {
    std::unique_lock<std::mutex> lock(main_fifo_mu_);
    for (const auto& meta : chunk_metas) {
      if (!meta.HasKey("__name")) {
        continue;
      }
      const auto& chunk_name = meta.GetKeyValue<std::string>("__name");
      const auto chunk_id = meta.GetId();
      auto it = main_fifo_.findWithoutPromotion(chunk_name);
      if (it == main_fifo_.end()) {
        FifoEntry entry;
        entry.object_id = chunk_id;
        // For insert here, we use a dedicate prune hook to remove
        // delete operations from the critical path
        VLOG(100) << "Main fifo: insert " << chunk_name << ", obj id "
                  << ObjectIDToString(chunk_id);
        main_fifo_.set(chunk_name, std::move(entry), /* promote */ false,
                       build_prune_hook);
      } else if (it->second.chunk_builder &&
                 it->second.chunk_builder->IsReady()) {
        // try to update access time
        auto access_time_label = meta.Label(KVCacheChunk::kFieldNameAccessTime);
        if (access_time_label.empty()) {
          access_time_label = std::to_string(
              meta.GetKeyValue<uint64_t>(KVCacheChunk::kFieldNameAccessTime));
        }
        uint64_t time = std::stoull(access_time_label);
        it->second.chunk_builder->SetGlobalAccessTime(time);
        it->second.chunk_builder->SetAccessTime(time);
      }
    }
  }

  if (!delete_list.empty()) {
    Delete(delete_list);
  }

  return Status::OK();
}

Status AIBrixBlobStorage::GetTokenChunkHashes(
    const std::vector<int>& prefix, const std::vector<int>& tokens,
    std::vector<std::string>& chunk_hashes) {
  std::vector<int> all(prefix.begin(), prefix.end());
  all.insert(all.end(), tokens.begin(), tokens.end());

  RETURN_ON_ERROR(
      hasher_->computeChunkHashesForTokens(all, chunk_size_, chunk_hashes));
  auto sz = tokens.size() / chunk_size_;
  chunk_hashes =
      std::vector<std::string>(chunk_hashes.end() - sz, chunk_hashes.end());
  return Status::OK();
}

#define DEFINE_TASK_FN(FN, OP, CB)                                        \
  auto FN = [this, &prefix, &tokens, &kv_tensors, cb = CB](               \
                size_t i,                                                 \
                std::shared_ptr<KVCacheChunkBuilder> builder) -> Status { \
    auto chunk_size = this->chunk_size_;                                  \
    if (builder == nullptr) {                                             \
      return Status::OK();                                                \
    }                                                                     \
                                                                          \
    std::vector<int> my_prefix(prefix.begin(), prefix.end());             \
    if (i > 0) {                                                          \
      my_prefix.insert(my_prefix.end(), tokens.begin(),                   \
                       tokens.begin() + i * chunk_size);                  \
    }                                                                     \
    std::vector<int> my_tokens(tokens.begin() + i * chunk_size,           \
                               tokens.begin() + (i + 1) * chunk_size);    \
                                                                          \
    std::vector<std::vector<std::pair<LLMKV, LLMKV>>> my_kv_tensors(      \
        kv_tensors.begin() + i * chunk_size,                              \
        kv_tensors.begin() + (i + 1) * chunk_size);                       \
                                                                          \
    auto status = builder->OP(my_prefix, my_tokens, my_kv_tensors);       \
    if (status.ok()) {                                                    \
      cb(i, my_kv_tensors);                                               \
    }                                                                     \
    return status;                                                        \
  }

#define WAIT_TASK_RESULTS(TIDS, COUNTER, FIRST_ERROR, OBJ_NAMES) \
  {                                                              \
    bool skip_rest = false;                                      \
    for (size_t i = 0; i < TIDS.size(); ++i) {                   \
      auto status = tg.TaskResult(TIDS[i]);                      \
      if (status.ok() && !skip_rest) {                           \
        COUNTER += chunk_size_;                                  \
      } else if (!status.ok() && skip_rest == false) {           \
        FIRST_ERROR = status;                                    \
        skip_rest = true;                                        \
        VLOG(100) << "First error: " << FIRST_ERROR.ToString();  \
      } else {                                                   \
        /* delete from index */                                  \
        {                                                        \
          std::unique_lock<std::mutex> lock(main_fifo_mu_);      \
          main_fifo_.erase(OBJ_NAMES[i]);                        \
        }                                                        \
        {                                                        \
          std::unique_lock<std::mutex> lock(small_fifo_mu_);     \
          small_fifo_.erase(OBJ_NAMES[i]);                       \
        }                                                        \
        VLOG(100) << "Error: " << status.ToString();             \
      }                                                          \
    }                                                            \
  }

Status AIBrixBlobStorage::UpdateInternal(
    const std::vector<int>& prefix, const std::vector<int>& tokens,
    const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kv_tensors,
    size_t& updated) {
  updated = 0;

  if (exit_flag_) {
    return Status::Invalid("Storage has been closed");
  }

  if (prefix.size() % chunk_size_ != 0) {
    return Status::Invalid("Prefix size " + std::to_string(prefix.size()) +
                           " should be multiple of chunk size " +
                           std::to_string(chunk_size_));
  }

  if (tokens.size() != kv_tensors.size()) {
    return Status::Invalid("Tokens size " + std::to_string(tokens.size()) +
                           " should be equal to kv tensors size " +
                           std::to_string(kv_tensors.size()));
  }

  if (tokens.size() > kMaxTokensPerSeq) {
    return Status::Invalid("Token list size exceeds the size limit");
  }

  std::vector<std::string> chunk_hashes;
  GetTokenChunkHashes(prefix, tokens, chunk_hashes);
  std::vector<std::string> obj_names;
  for (const auto& chunk_hash : chunk_hashes) {
    obj_names.push_back(kv_cache_ns_ + chunk_hash);
  }

  auto now = std::chrono::system_clock::now().time_since_epoch();
  auto access_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();

  parallel::ThreadGroup tg(
      std::min(obj_names.size(),
               static_cast<size_t>(std::thread::hardware_concurrency())));
  std::vector<parallel::ThreadGroup::tid_t> tids;

  DEFINE_TASK_FN(fn, Update, [](auto, auto&&) {});

  auto ret = Status::OK();
  size_t index;
  {
    std::unique_lock<std::mutex> slock(small_fifo_mu_);
    std::unique_lock<std::mutex> mlock(main_fifo_mu_);
    for (index = 0; index < obj_names.size(); index++) {
      const auto& obj_name = obj_names[index];
      auto sit = small_fifo_.findWithoutPromotion(obj_name);
      if (sit == small_fifo_.end()) {
        auto mit = main_fifo_.findWithoutPromotion(obj_name);
        if (mit == main_fifo_.end()) {
          break;
        }
      }
    }
  }

  updated += index * chunk_size_;

  if (index == obj_names.size()) {
    // we find all chunks in local
    return ret;
  }

  // right now index points to the first missing chunk
  for (size_t i = index; i < obj_names.size(); i++) {
    const auto& obj_name = obj_names[i];
    EvictingCacheMap<std::string, FifoEntry>* target_fifo = nullptr;
    std::mutex* target_fifo_mu = nullptr;

    {
      std::unique_lock<std::mutex> lock(ghost_fifo_mu_);
      auto it = ghost_fifo_.findWithoutPromotion(obj_name);
      if (it != ghost_fifo_.end()) {
        target_fifo = &main_fifo_;
        target_fifo_mu = &main_fifo_mu_;
      } else {
        target_fifo = &small_fifo_;
        target_fifo_mu = &small_fifo_mu_;
      }
    }

    FifoEntry entry;
    {
      std::unique_lock<std::mutex> lock(*target_fifo_mu);
      auto it = target_fifo->findWithoutPromotion(obj_name);
      if (it != target_fifo->end()) {
        // chunk is already in the cache
        tids.push_back(tg.AddTask(fn, i, nullptr));
        continue;
      }

      auto status = KVCacheChunkBuilder::Make(
          entry.chunk_builder, rpc_client_, kMaxTokensPerSeq, tensor_nbytes_,
          layer_, chunk_size_, kv_cache_ns_);
      if (!status.ok()) {
        VLOG(100) << "Failed to make chunk builder, " << status.ToString();
        ret += status;
        // skip this and rest chunks
        break;
      }

      entry.chunk_builder->SetAccessTime(access_time);
      if (VLOG_IS_ON(100)) {
        if (target_fifo == &main_fifo_) {
          LOG(INFO) << "Main fifo: insert " << obj_name;
        } else {
          LOG(INFO) << "Small fifo: insert " << obj_name;
        }
      }
      target_fifo->set(obj_name, entry, /* promote */ false);

      tids.push_back(tg.AddTask(fn, i, entry.chunk_builder));
    }
  }

  if (tids.empty()) {
    return ret;
  }

  Status first_error = Status::OK();
  WAIT_TASK_RESULTS(tids, updated, first_error, obj_names);
  return first_error;
}

Status AIBrixBlobStorage::QueryInternal(
    const std::vector<int>& prefix, const std::vector<int>& tokens,
    std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kv_tensors,
    size_t& matched) {
  matched = 0;

  if (exit_flag_) {
    return Status::Invalid("Storage has been closed");
  }

  if (prefix.size() % chunk_size_ != 0) {
    return Status::Invalid("Prefix size " + std::to_string(prefix.size()) +
                           " should be multiple of chunk size " +
                           std::to_string(chunk_size_));
  }

  if (tokens.size() != kv_tensors.size()) {
    return Status::Invalid("Tokens size " + std::to_string(tokens.size()) +
                           " should be equal to kv tensors size " +
                           std::to_string(kv_tensors.size()));
  }

  if (tokens.size() > kMaxTokensPerSeq) {
    return Status::Invalid("Token list size exceeds the size limit");
  }

  std::vector<std::string> chunk_hashes;
  GetTokenChunkHashes(prefix, tokens, chunk_hashes);
  std::vector<std::string> obj_names;
  for (const auto& chunk_hash : chunk_hashes) {
    obj_names.push_back(kv_cache_ns_ + chunk_hash);
  }

  auto now = std::chrono::system_clock::now().time_since_epoch();
  auto access_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();

  parallel::ThreadGroup tg(
      std::min(obj_names.size(),
               static_cast<size_t>(std::thread::hardware_concurrency())));
  std::vector<parallel::ThreadGroup::tid_t> tids;

  bool is_zero_copy = kv_tensors[0][0].first.data == nullptr;

  auto cb = [&kv_tensors, chunk_size = chunk_size_, is_zero_copy](
                size_t i, auto&& my_kv_tensors) {
    if (!is_zero_copy) {
      return;
    }

    // for zero-copy use case, we need to copy descriptors back
    size_t j = 0;
    for (auto it = kv_tensors.begin() + i * chunk_size;
         it != kv_tensors.begin() + (i + 1) * chunk_size; it++) {
      *it = my_kv_tensors[j++];
    }
  };

  DEFINE_TASK_FN(fn, Query, cb);

  for (size_t i = 0; i < obj_names.size(); i++) {
    const auto& obj_name = obj_names[i];

    std::shared_ptr<KVCacheChunkBuilder> query_builder = nullptr;
    // Check if the chunk is in S first, and then check M
    {
      std::unique_lock<std::mutex> lock(small_fifo_mu_);
      auto it = small_fifo_.findWithoutPromotion(obj_name);
      if (it != small_fifo_.end()) {
        VLOG(100) << "Hit " << obj_name << " in S";
        it->second.access_bit = true;
        it->second.chunk_builder->SetAccessTime(access_time);
        query_builder = it->second.chunk_builder;
      }
    }

    if (query_builder == nullptr) {
      std::unique_lock<std::mutex> lock(main_fifo_mu_);
      auto it = main_fifo_.findWithoutPromotion(obj_name);
      if (it != main_fifo_.end()) {
        VLOG(100) << "Hit " << obj_name << " in M";
        if (it->second.chunk_builder == nullptr) {
          VLOG(100) << "Loading " << obj_name;
          VINEYARD_ASSERT(it->second.object_id != InvalidObjectID());

          auto status = KVCacheChunkBuilder::Make(
              it->second.chunk_builder, rpc_client_, tensor_nbytes_, layer_,
              chunk_size_, kv_cache_ns_, it->second.object_id);
          if (!status.ok()) {
            VLOG(100) << "Failed to make chunk builder, " << status.ToString();
            // skip this and rest chunks
            break;
          } else {
            VLOG(100) << "obj name=" << obj_name
                      << ", obj id=" << ObjectIDToString(it->second.object_id);
          }
        }
        it->second.access_bit = true;
        it->second.chunk_builder->SetAccessTime(access_time);
        query_builder = it->second.chunk_builder;
      }
    }

    // cache miss
    if (query_builder == nullptr) {
      break;
    }

    // cache hit
    tids.push_back(tg.AddTask(fn, i, query_builder));
  }

  if (tids.empty()) {
    return Status::ObjectNotExists();
  }

  Status first_error = Status::OK();
  WAIT_TASK_RESULTS(tids, matched, first_error, obj_names);
  return first_error;
}

Status AIBrixBlobStorage::Update(
    const std::vector<int>& tokens,
    const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kv_tensors,
    size_t& updated) {
  return UpdateInternal({}, tokens, kv_tensors, updated);
}

Status AIBrixBlobStorage::Update(
    const std::vector<int>& prefix, const std::vector<int>& tokens,
    const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kv_tensors,
    size_t& updated) {
  return UpdateInternal(prefix, tokens, kv_tensors, updated);
}

Status AIBrixBlobStorage::Query(
    const std::vector<int>& tokens,
    std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kv_tensors,
    size_t& matched) {
  return QueryInternal({}, tokens, kv_tensors, matched);
}

Status AIBrixBlobStorage::Query(
    const std::vector<int>& prefix, const std::vector<int>& tokens,
    std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kv_tensors,
    size_t& matched) {
  return QueryInternal(prefix, tokens, kv_tensors, matched);
}

Status AIBrixBlobStorage::SealAndPersist(
    const std::string& name,
    const std::shared_ptr<KVCacheChunkBuilder>& chunk_builder,
    ObjectID& chunk_id) {
  auto& client = GetClient();
  std::shared_ptr<Object> chunk = chunk_builder->Seal();
  if (chunk == nullptr) {
    return Status::IOError();
  }
  chunk_id = chunk->id();
  RETURN_ON_ERROR(client.Persist(chunk_id));
  VINEYARD_DISCARD(
      client.Label(chunk_id, KVCacheChunk::kFieldNameAccessTime,
                   std::to_string(chunk_builder->GetAccessTime())));
  auto status = client.PutName(chunk_id, name, /* unique */ true);
  if (status.IsNameExists()) {
    ObjectID obj_id;
    auto s = client.GetName(name, obj_id);
    if (!s.ok()) {
      VLOG(100) << "Failed to get obj id of existing chunk, name=" << name
                << ", error=" << s.ToString();
    } else {
      ObjectMeta meta;
      s = client.GetMetaData(obj_id, meta);

      if (!s.ok()) {
        VLOG(100) << "Failed to get meta of existing chunk, name=" << name
                  << ", id=" << ObjectIDToString(obj_id)
                  << ", error=" << s.ToString();
      } else {
        if (chunk_builder->IsSame(meta).ok()) {
          VLOG(100)
              << "Existing chunk is the same as the persisting chunk, name="
              << name << ", obj ids: " << ObjectIDToString(chunk_id) << ", "
              << ObjectIDToString(obj_id);
          VINEYARD_DISCARD(client.DelData(chunk_id));
          // reuse existing chunk
          chunk_id = obj_id;
          return Status::OK();
        } else {
          VLOG(100) << "A different chunk has taken name=" << name
                    << ", obj id=" << ObjectIDToString(obj_id);
          // go to the following if branch
        }
      }
    }
  }

  if (!status.ok()) {
    VLOG(100) << "Failed to put name " << name << ", " << status.ToString();
    // Just delete chunk id and wait for the next time
    VINEYARD_DISCARD(client.DelData(chunk_id));
  }
  return status;
}

Status AIBrixBlobStorage::Delete(const std::vector<std::string>& chunk_list) {
  Status status = Status::OK();
  std::vector<ObjectID> delete_ids;
  auto& client = GetClient();
  for (const auto& name : chunk_list) {
    ObjectID id;
    if (client.GetName(name, id, false).ok()) {
      delete_ids.push_back(id);
      status += client.DropName(name);
    } else {
      VLOG(100) << "Failed to get obj id for name=" << name;
    }
  }

  status += Delete(delete_ids);

  return status;
}

Status AIBrixBlobStorage::Delete(const std::vector<ObjectID>& obj_ids) {
  Status status = Status::OK();
  if (obj_ids.size() > 0) {
    auto& client = GetClient();
    status += client.DelData(obj_ids);
    if (status.ok()) {
      if (VLOG_IS_ON(100)) {
        for (auto id : obj_ids) {
          LOG(INFO) << "Deleted obj " << ObjectIDToString(id);
        }
      }
    } else {
      if (VLOG_IS_ON(100)) {
        for (auto id : obj_ids) {
          LOG(INFO) << "Failed to delete obj " << ObjectIDToString(id) << ", "
                    << status.ToString();
        }
      }
    }
  }
  return status;
}

void AIBrixBlobStorage::CloseCache() {
  LOG(INFO) << "Close AIBrixBlobStorage";
  TerminateGCThreads();
}

std::string AIBrixBlobStorage::GetTimestamp(
    std::chrono::duration<int64_t, std::nano> time) {
  auto duration_since_epoch =
      std::chrono::duration_cast<std::chrono::system_clock::duration>(time);
  std::chrono::time_point<std::chrono::system_clock> timestamp =
      std::chrono::system_clock::time_point(duration_since_epoch);
  time_t t = std::chrono::system_clock::to_time_t(timestamp);

  std::tm tm;
  localtime_r(&t, &tm);
  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
  return oss.str();
}

Status AIBrixBlobStorage::ListKVCache(const std::string& prefix,
                                      std::vector<ObjectMeta>& metas) {
  std::string ns = KVCacheChunk::GetNameSpace(prefix);
  std::vector<ObjectID> ids;
  auto status = rpc_client_.ListBy(KVCacheChunk::kFieldNameNS, ns, false,
                                   UINT64_MAX, ids);
  if (!status.ok()) {
    VLOG(100) << "Failed to list by namespace=" << ns
              << ", error=" << status.ToString();
    return status;
  }

  return rpc_client_.GetMetaData(ids, metas, /* sync remote */ true);
}

Status AIBrixBlobStorage::ProcessPersistList(
    const std::vector<std::pair<std::string, FifoEntry>>& persist_list) {
  VLOG(100) << "ProcessPersistList: #persist chunks=" << persist_list.size();

  auto& client = GetClient();
  for (const auto& pair : persist_list) {
    auto& key = pair.first;
    auto& chunk_builder = pair.second.chunk_builder;
    ObjectID chunk_id;
    auto s = SealAndPersist(key, chunk_builder, chunk_id);
    if (s.ok()) {
      VLOG(100) << "Persist " << key
                << ", obj id=" << ObjectIDToString(chunk_id);
      std::unique_lock<std::mutex> lock(main_fifo_mu_);
      auto it = main_fifo_.findWithoutPromotion(key);
      if (it != main_fifo_.end()) {
        VLOG(100) << "Main fifo: " << key
                  << " obj id=" << ObjectIDToString(chunk_id);
        it->second.object_id = chunk_id;
      } else {
        VLOG(100) << "Main fifo: " << key << " not in fifo, delete obj "
                  << chunk_id;
        VINEYARD_DISCARD(client.DelData(chunk_id));
      }
    } else {
      VLOG(100) << "Failed to seal and persist " << key
                << ", error=" << s.ToString();
    }
  }

  return Status::OK();
}

Status AIBrixBlobStorage::ProcessUpdateList(
    const std::vector<std::pair<std::string, FifoEntry>>& update_list) {
  VLOG(100) << "ProcessUpdateList: #update chunks=" << update_list.size();

  auto& client = GetClient();
  for (const auto& pair : update_list) {
    auto& key = pair.first;
    auto& chunk_builder = pair.second.chunk_builder;
    auto& chunk_id = pair.second.object_id;

    auto update_status =
        client.Label(chunk_id, KVCacheChunk::kFieldNameAccessTime,
                     std::to_string(chunk_builder->GetAccessTime()));

    chunk_builder->SetGlobalAccessTime(chunk_builder->GetAccessTime());

    if (update_status.ok()) {
      VLOG(100) << "Updated " << key << "'s access time to "
                << GetTimestamp(std::chrono::nanoseconds(
                       chunk_builder->GetAccessTime()));
    } else {
      VLOG(100) << "Failed to update " << key << "'s access time"
                << ", error=" << update_status.ToString();
    }
  }

  return Status::OK();
}

Status AIBrixBlobStorage::LocalSyncFunc() {
  // load global main fifo and merge it with the local one
  VINEYARD_DISCARD(BuildMainFifo());

  using PairT = std::pair<std::string, FifoEntry>;
  std::vector<PairT> persist_list;
  std::vector<PairT> update_list;
  {
    std::unique_lock<std::mutex> lock(main_fifo_mu_);
    for (auto& pair : main_fifo_) {
      auto& key = pair.first;
      auto& entry = pair.second;
      if (!entry.chunk_builder || !entry.chunk_builder->IsReady()) {
        // skip never accessed chunks
        // skip not ready chunks
        continue;
      }

      if (entry.object_id == InvalidObjectID()) {
        persist_list.push_back({key, entry});
      }

      if (entry.object_id != InvalidObjectID() &&
          entry.chunk_builder->GetAccessTime() >
              entry.chunk_builder->GetGlobalAccessTime() +
                  local_sync_interval_s_.count() * 1000000000) {
        update_list.push_back({key, entry});
      }
    }
  }

  auto status = ProcessPersistList(persist_list);

  status += ProcessUpdateList(update_list);

  return status;
}

Status AIBrixBlobStorage::GlobalGCFunc() {
  auto now = std::chrono::high_resolution_clock::now();
  auto nanoseconds_since_epoch =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          now.time_since_epoch());
  std::vector<ObjectMeta> chunk_metas;
  std::vector<std::string> delete_chunks;
  std::vector<ObjectID> delete_list;
  RETURN_ON_ERROR(ListKVCache(kv_cache_ns_, chunk_metas));
  VLOG(100) << "Global GC: " << chunk_metas.size() << " chunks to check";
  for (const auto& meta : chunk_metas) {
    const auto chunk_id = meta.GetId();
    std::string chunk_name("unknown");
    if (meta.HasKey("__name")) {
      chunk_name = meta.GetKeyValue<std::string>("__name");
    }
    auto access_time_label = meta.Label(KVCacheChunk::kFieldNameAccessTime);
    if (access_time_label.empty()) {
      access_time_label = std::to_string(
          meta.GetKeyValue<uint64_t>(KVCacheChunk::kFieldNameAccessTime));
    }
    uint64_t time = std::stoull(access_time_label);
    auto access_time = std::chrono::nanoseconds(time);
    VLOG(100) << "Chunk TTL: " << global_ttl_s_.count() << " s";
    if ((access_time + global_ttl_s_).count() <
        nanoseconds_since_epoch.count()) {
      VLOG(100) << "Global GC: " << chunk_name << " is GC'ed";
      VLOG(100) << "Access time: " << GetTimestamp(access_time);
      VLOG(100) << "Now: " << GetTimestamp(nanoseconds_since_epoch);
      delete_chunks.emplace_back(chunk_name);
      delete_list.emplace_back(chunk_id);
    } else {
      VLOG(100) << "Global GC: " << chunk_name << " is alive";
      VLOG(100) << "Access time: " << GetTimestamp(access_time);
      VLOG(100) << "Now: " << GetTimestamp(nanoseconds_since_epoch);
    }
  }

  if (delete_list.size() > 0) {
    {
      std::unique_lock<std::mutex> lock(main_fifo_mu_);
      for (const auto& name : delete_chunks) {
        main_fifo_.erase(name);
      }
    }
    VINEYARD_DISCARD(Delete(delete_list));
  }
  return Status::OK();
}

#define DEFINE_GC_THREAD(NAME, ENABLED, GC_MU, GC_CV, GC_INTERVAL)             \
  void AIBrixBlobStorage::NAME##Thread(                                        \
      std::shared_ptr<AIBrixBlobStorage> self) {                               \
    int64_t last_time =                                                        \
        std::chrono::duration_cast<std::chrono::seconds>(                      \
            std::chrono::high_resolution_clock::now().time_since_epoch())      \
            .count();                                                          \
    while (1) {                                                                \
      std::unique_lock<std::mutex> lock(self->GC_MU);                          \
      auto interval = self->GC_INTERVAL;                                       \
      if (self->GC_CV.wait_for(lock, interval, [self, &last_time, &interval] { \
            int64_t current_time =                                             \
                std::chrono::duration_cast<std::chrono::seconds>(              \
                    std::chrono::high_resolution_clock::now()                  \
                        .time_since_epoch())                                   \
                    .count();                                                  \
            return self->exit_flag_ ||                                         \
                   (current_time - last_time) > interval.count();              \
          })) {                                                                \
        if (!(self->ENABLED)) {                                                \
          LOG(INFO) << #NAME " skipped";                                       \
          return;                                                              \
        }                                                                      \
        if (self->exit_flag_) {                                                \
          LOG(INFO) << #NAME " exit";                                          \
          return;                                                              \
        }                                                                      \
        LOG(INFO) << #NAME " started";                                         \
        Status status = self->NAME##Func();                                    \
        if (!status.ok()) {                                                    \
          LOG(ERROR) << #NAME " failed: " << status.ToString();                \
          /* Not a fatal error and wait for next time */                       \
        } else {                                                               \
          LOG(INFO) << #NAME " completed";                                     \
        }                                                                      \
        last_time = std::chrono::duration_cast<std::chrono::seconds>(          \
                        std::chrono::system_clock::now().time_since_epoch())   \
                        .count();                                              \
      }                                                                        \
    }                                                                          \
  }

DEFINE_GC_THREAD(LocalSync, local_gc_enabled_, local_sync_mu_, local_sync_cv_,
                 local_sync_interval_s_);
DEFINE_GC_THREAD(GlobalGC, global_gc_enabled_, global_gc_mu_, global_gc_cv_,
                 global_gc_interval_s_);

void AIBrixBlobStorage::TerminateGCThreads() {
  std::lock_guard<std::mutex> local_lock(local_sync_mu_);
  std::lock_guard<std::mutex> global_lock(global_gc_mu_);
  if (!exit_flag_) {
    exit_flag_ = true;

    VLOG(100) << "Terminating global GC thread";
    global_gc_mu_.unlock();
    global_gc_cv_.notify_all();
    global_gc_thread_.join();

    VLOG(100) << "Terminating local sync thread";
    local_sync_mu_.unlock();
    local_sync_cv_.notify_all();
    local_sync_thread_.join();
  }
}

ClientBase& AIBrixBlobStorage::GetClient() { return rpc_client_; }

}  // namespace vineyard
