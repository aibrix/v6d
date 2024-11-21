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
#include <iomanip>
#include <memory>
#include <string>
#include <utility>

#include "client/client.h"
#include "common/memory/memcpy.h"
#include "common/util/logging.h"
#include "llm-cache/ds/kv_cache_block.h"  // LLMKV
#include "llm-cache/ds/kv_cache_chunk.h"
#include "llm-cache/hash/md5.h"

namespace vineyard {

void KVCacheChunk::Construct(const ObjectMeta& meta) {
  Object::Construct(meta);

  std::string tname = type_name<KVCacheChunk>();

  VINEYARD_ASSERT(
      meta.GetTypeName() == tname,
      "Expect typename '" + tname + "', but got '" + meta.GetTypeName() + "'");

  // 1. construct the member field
  total_tokens_ = meta.GetKeyValue<int>(KVCacheChunk::kFieldNameTotalTokens);
  tensor_nbytes_ = meta.GetKeyValue<int>(KVCacheChunk::kFieldNameTensorNBytes);
  layer_ = meta.GetKeyValue<int>(KVCacheChunk::kFieldNameLayer);
  chunk_size_ = meta.GetKeyValue<int>(KVCacheChunk::kFieldNameChunkSize);
  access_time_ =
      meta_.GetKeyValue<uint64_t>(KVCacheChunk::kFieldNameAccessTime);
  md5_ = meta_.GetKeyValue<std::string>(KVCacheChunk::kFieldNameMd5);
  ns_ = meta_.GetKeyValue<std::string>(KVCacheChunk::kFieldNameNS);

  // 2. construct the buffer
  ObjectMeta blob_meta;
  meta_.GetMemberMeta(KVCacheChunk::kFieldNameBuffer, blob_meta);
  ObjectID blob_id = blob_meta.GetId();
  meta.GetBuffer(blob_id, buffer_);
}

Status KVCacheChunkBuilder::Make(std::shared_ptr<KVCacheChunkBuilder>& builder,
                                 RPCClient& rpc_client, int max_tokens,
                                 int tensor_nbytes, int layer, int chunk_size,
                                 const std::string& kv_cache_ns) {
  size_t size = static_cast<size_t>(chunk_size) * layer * tensor_nbytes * 2 +
                max_tokens * sizeof(int);
  builder = std::make_shared<KVCacheChunkBuilder>(
      rpc_client, tensor_nbytes, layer, chunk_size, kv_cache_ns);
  builder->chunk_id_ = InvalidObjectID();
  builder->remote_buffer_writer_ = std::make_shared<RemoteBlobWriter>(size);

  auto now = std::chrono::system_clock::now().time_since_epoch();
  builder->g_access_time_ =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();

  return Status::OK();
}

Status KVCacheChunkBuilder::Make(std::shared_ptr<KVCacheChunkBuilder>& builder,
                                 RPCClient& rpc_client, int tensor_nbytes,
                                 int layer, int chunk_size,
                                 const std::string& kv_cache_ns,
                                 ObjectID chunk_id) {
  builder = std::make_shared<KVCacheChunkBuilder>(
      rpc_client, tensor_nbytes, layer, chunk_size, kv_cache_ns);
  builder->chunk_id_ = chunk_id;
  return Status::OK();
}

Status KVCacheChunkBuilder::IsSame(const ObjectMeta& meta) {
  RETURN_ON_ASSERT(is_ready_);
  if (VLOG_IS_ON(100)) {
    LOG(INFO) << "this: {total_tokens=" << total_tokens_
              << ", chunk_size=" << chunk_size_
              << ", tensor_nbytes=" << tensor_nbytes_ << ", layer=" << layer_
              << ", md5=" << md5_ << "}";
    LOG(INFO) << "that: " << meta.ToString();
  }

  RETURN_ON_ASSERT(meta.HasKey(KVCacheChunk::kFieldNameTotalTokens));
  RETURN_ON_ASSERT(meta.HasKey(KVCacheChunk::kFieldNameChunkSize));
  RETURN_ON_ASSERT(meta.HasKey(KVCacheChunk::kFieldNameTensorNBytes));
  RETURN_ON_ASSERT(meta.HasKey(KVCacheChunk::kFieldNameLayer));
  RETURN_ON_ASSERT(meta.HasKey(KVCacheChunk::kFieldNameAccessTime));
  RETURN_ON_ASSERT(meta.HasKey(KVCacheChunk::kFieldNameMd5));
  RETURN_ON_ASSERT(meta.HasKey(KVCacheChunk::kFieldNameNS));

  RETURN_ON_ASSERT(meta.GetKeyValue<int>(KVCacheChunk::kFieldNameTotalTokens) ==
                   total_tokens_);
  RETURN_ON_ASSERT(meta.GetKeyValue<int>(KVCacheChunk::kFieldNameChunkSize) ==
                   chunk_size_);
  RETURN_ON_ASSERT(meta.GetKeyValue<int>(
                       KVCacheChunk::kFieldNameTensorNBytes) == tensor_nbytes_);
  RETURN_ON_ASSERT(meta.GetKeyValue<int>(KVCacheChunk::kFieldNameLayer) ==
                   layer_);
  // We assume it's not possilbe to have same name and md5 of all tokens
  RETURN_ON_ASSERT(meta.GetKeyValue<std::string>(KVCacheChunk::kFieldNameMd5) ==
                   md5_);

  RETURN_ON_ASSERT(meta.GetKeyValue<std::string>(KVCacheChunk::kFieldNameNS) ==
                   ns_);

  return Status::OK();
}

Status KVCacheChunkBuilder::Query(
    const std::vector<int>& prefix, const std::vector<int>& tokens,
    std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kv_tensor) {
  return QueryImpl(prefix, tokens, kv_tensor);
}

Status KVCacheChunkBuilder::Construct() {
  ObjectMeta object_meta;
  std::shared_ptr<Object> object = nullptr;
  std::shared_ptr<KVCacheChunk> chunk = nullptr;

  RETURN_ON_ASSERT(rpc_client_.Connected());

  VLOG(100) << "Constructing " << ObjectIDToString(chunk_id_);
  auto status = Status::OK();
  status = rpc_client_.GetMetaData(chunk_id_, object_meta, true);
  if (!status.ok()) {
    VLOG(100) << "Get meta data failed: " << status.ToString();
    return Status::ObjectNotExists();
  } else if (object_meta.IsLocal()) {
    object = rpc_client_.GetObject(chunk_id_);
  }

  // fetch from remote
  if (object == nullptr) {
    std::map<InstanceID, json> cluster_info;
    RETURN_ON_ERROR(rpc_client_.ClusterInfo(cluster_info));
    std::string rpc_endpoint =
        cluster_info[object_meta.GetInstanceId()].value("rpc_endpoint", "");
    if (!rpc_endpoint.empty()) {
      std::string rdma_endpoint =
          cluster_info[object_meta.GetInstanceId()].value("rdma_endpoint", "");
      RPCClient remote_rpc_client;
      RETURN_ON_ERROR(
          remote_rpc_client.Connect(rpc_endpoint, "", "", rdma_endpoint));
      object = remote_rpc_client.GetObject(chunk_id_);
      RETURN_ON_ASSERT(object != nullptr);
      ObjectID buffer_id =
          object_meta.GetMember(KVCacheChunk::kFieldNameBuffer)->id();
      std::shared_ptr<RemoteBlob> blob;
      RETURN_ON_ERROR(remote_rpc_client.GetRemoteBlob(buffer_id, blob));
      std::dynamic_pointer_cast<KVCacheChunk>(object)->buffer_ = blob->Buffer();
    }
  }

  RETURN_ON_ASSERT(object != nullptr, "object is nullptr");
  LOG(INFO) << "Got " << ObjectIDToString(chunk_id_) << " from instance "
            << object_meta.GetInstanceId();

  chunk = std::dynamic_pointer_cast<KVCacheChunk>(object);
  if (chunk->buffer_ == nullptr) {
    return Status::IOError();
  }

  if (chunk_id_ != chunk->id()) {
    // If the object is migrated, we should delete the copied object.
    status = rpc_client_.DelData(chunk->id());
    if (!status.ok()) {
      LOG(ERROR) << "Delete object failed: " << status.ToString()
                 << " It may cause memory leak.";
    }
  }

  // sanity check
  RETURN_ON_ASSERT(tensor_nbytes_ == chunk->tensor_nbytes_);
  RETURN_ON_ASSERT(layer_ == chunk->layer_);
  RETURN_ON_ASSERT(chunk_size_ == chunk->chunk_size_);
  RETURN_ON_ASSERT(ns_ == chunk->ns_);

  // construct meta info
  auto all_tokens_off =
      chunk->chunk_size_ * chunk->layer_ * chunk->tensor_nbytes_ * 2;
  total_tokens_ = chunk->total_tokens_;
  g_access_time_ = chunk->access_time_;
  buffer_ = chunk->buffer_;

  auto* buffer = buffer_->data();

  all_tokens_.resize(chunk->total_tokens_);
  std::memcpy(all_tokens_.data(), buffer + all_tokens_off,
              chunk->total_tokens_ * sizeof(int));

  md5_ = md5(std::string(all_tokens_.begin(), all_tokens_.end()));
  RETURN_ON_ASSERT(md5_ == chunk->md5_);

  return Status::OK();
}

Status KVCacheChunkBuilder::QueryImpl(
    const std::vector<int>& prefix, const std::vector<int>& tokens,
    std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kv_tensor) {
  RETURN_ON_ASSERT(tokens.size() == chunk_size_,
                   "The size of tokens is not equal to chunk_size");
  RETURN_ON_ASSERT(kv_tensor.size() == chunk_size_,
                   "The size of kv tensor is not equal to chunk_size");
  if (!is_ready_) {
    std::unique_lock<std::mutex> wlock(mutex_);
    if (!is_ready_ && chunk_id_ != InvalidObjectID()) {
      // need to construct from given chunk
      Construct();
      is_ready_ = true;
      cv_.notify_all();
    } else if (!is_ready_) {
      // need to wait for the completion of update
      cv_.wait(wlock, [this]() -> bool { return this->is_ready_; });
    }
  }

  return QueryInternal(prefix, tokens, kv_tensor);
}

Status KVCacheChunkBuilder::QueryInternal(
    const std::vector<int>& prefix, const std::vector<int>& tokens,
    std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kv_tensor) {
  uint8_t* buffer = nullptr;
  if (remote_buffer_writer_ != nullptr) {
    buffer = reinterpret_cast<uint8_t*>(remote_buffer_writer_->data());
  } else if (buffer_ != nullptr) {
    buffer = const_cast<uint8_t*>(buffer_->data());
  }

  RETURN_ON_ASSERT(buffer != nullptr, "failed chunk");

  auto all_tokens = prefix;
  all_tokens.insert(all_tokens.end(), tokens.begin(), tokens.end());
  if (VLOG_IS_ON(100) && (all_tokens != all_tokens_)) {
    auto str0 = "all_tokens[" + std::to_string(all_tokens.size()) + "]: ";
    for (int i = 0; i < all_tokens.size(); i++) {
      str0 += std::to_string(all_tokens[i]) + ", ";
    }
    auto str1 = "all_tokens_[" + std::to_string(all_tokens_.size()) + "]: ";
    for (int i = 0; i < all_tokens_.size(); i++) {
      str1 += std::to_string(all_tokens_[i]) + ", ";
    }
    LOG(INFO) << str0;
    LOG(INFO) << str1;
  }
  RETURN_ON_ASSERT(all_tokens == all_tokens_, "tokens not match");

  if (kv_tensor[0][0].first.data == nullptr) {
    for (int i = 0; i < chunk_size_; i++) {
      VINEYARD_ASSERT(kv_tensor[i].size() == layer_);
      uint8_t* key_tensor_chunk_data = buffer + i * layer_ * tensor_nbytes_;
      uint8_t* value_tensor_chunk_data =
          key_tensor_chunk_data + chunk_size_ * layer_ * tensor_nbytes_;

      for (int j = 0; j < layer_; j++) {
        LLMKV& key_tensor = kv_tensor[i][j].first;
        LLMKV& value_tensor = kv_tensor[i][j].second;
        VINEYARD_ASSERT(key_tensor.data == nullptr);
        VINEYARD_ASSERT(value_tensor.data == nullptr);

        key_tensor.data = key_tensor_chunk_data + j * tensor_nbytes_;
        key_tensor.length = tensor_nbytes_;
        value_tensor.data = value_tensor_chunk_data + j * tensor_nbytes_;
        value_tensor.length = tensor_nbytes_;
      }
    }
  } else {
    std::vector<void*> dst_buffers;
    std::vector<const void*> src_buffers;
    dst_buffers.reserve(chunk_size_ * layer_ * 2);
    src_buffers.reserve(chunk_size_ * layer_ * 2);

    for (int i = 0; i < chunk_size_; i++) {
      uint8_t* key_tensor_chunk_data = buffer + i * layer_ * tensor_nbytes_;

      for (int j = 0; j < layer_; j++) {
        LLMKV& key_tensor = kv_tensor[i][j].first;
        VINEYARD_ASSERT(key_tensor.data != nullptr);
        VINEYARD_ASSERT(key_tensor.length == tensor_nbytes_);

        uint8_t* key_tensor_data = key_tensor_chunk_data + j * tensor_nbytes_;

        dst_buffers.push_back(key_tensor.data);
        src_buffers.push_back(key_tensor_data);
      }
    }

    for (int i = 0; i < chunk_size_; i++) {
      uint8_t* key_tensor_chunk_data = buffer + i * layer_ * tensor_nbytes_;
      uint8_t* value_tensor_chunk_data =
          key_tensor_chunk_data + chunk_size_ * layer_ * tensor_nbytes_;

      for (int j = 0; j < layer_; j++) {
        LLMKV& value_tensor = kv_tensor[i][j].second;
        VINEYARD_ASSERT(value_tensor.data != nullptr);
        VINEYARD_ASSERT(value_tensor.length == tensor_nbytes_);

        uint8_t* value_tensor_data =
            value_tensor_chunk_data + j * tensor_nbytes_;

        dst_buffers.push_back(value_tensor.data);
        src_buffers.push_back(value_tensor_data);
      }
    }

    vineyard::memory::concurrent_memcpy_n(dst_buffers, src_buffers,
                                          tensor_nbytes_);
  }

  if (VLOG_IS_ON(200)) {
    PrintKVCacheChunk();
  }
  return Status::OK();
}

Status KVCacheChunkBuilder::Update(
    const std::vector<int>& prefix, const std::vector<int>& tokens,
    const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kv_tensor) {
  return UpdateImpl(prefix, tokens, kv_tensor);
}

Status KVCacheChunkBuilder::UpdateImpl(
    const std::vector<int>& prefix, const std::vector<int>& tokens,
    const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kv_tensor) {
  VINEYARD_ASSERT(tokens.size() == chunk_size_,
                  "The size of tokens is not equal to chunk_size");
  VINEYARD_ASSERT(kv_tensor.size() == chunk_size_,
                  "The size of kv tensor is not equal to chunk_size");

  VINEYARD_ASSERT(remote_buffer_writer_ != nullptr,
                  "remote_buffer_writer_ is nullptr");

  VINEYARD_ASSERT(!is_ready_);

  std::unique_lock<std::mutex> wlock(mutex_);

  all_tokens_.reserve(prefix.size() + tokens.size());
  all_tokens_.insert(all_tokens_.end(), prefix.begin(), prefix.end());
  all_tokens_.insert(all_tokens_.end(), tokens.begin(), tokens.end());

  md5_ = md5(std::string(all_tokens_.begin(), all_tokens_.end()));

  total_tokens_ = all_tokens_.size();
  VINEYARD_ASSERT(total_tokens_ == prefix.size() + tokens.size());
  auto buffer = reinterpret_cast<uint8_t*>(remote_buffer_writer_->data());

  std::vector<void*> dst_buffers;
  std::vector<const void*> src_buffers;
  dst_buffers.reserve(chunk_size_ * layer_ * 2);
  src_buffers.reserve(chunk_size_ * layer_ * 2);

  for (int i = 0; i < chunk_size_; i++) {
    uint8_t* key_tensor_chunk_data = buffer + i * layer_ * tensor_nbytes_;
    VINEYARD_ASSERT(kv_tensor[i].size() == layer_);

    for (int j = 0; j < layer_; j++) {
      LLMKV key_tensor = kv_tensor[i][j].first;
      VINEYARD_ASSERT(key_tensor.length == tensor_nbytes_);

      uint8_t* key_tensor_data = key_tensor_chunk_data + j * tensor_nbytes_;
      dst_buffers.push_back(key_tensor_data);
      src_buffers.push_back(key_tensor.data);
    }
  }

  for (int i = 0; i < chunk_size_; i++) {
    uint8_t* key_tensor_chunk_data = buffer + i * layer_ * tensor_nbytes_;
    uint8_t* value_tensor_chunk_data =
        key_tensor_chunk_data + chunk_size_ * layer_ * tensor_nbytes_;

    for (int j = 0; j < layer_; j++) {
      LLMKV value_tensor = kv_tensor[i][j].second;
      VINEYARD_ASSERT(value_tensor.length == tensor_nbytes_);

      uint8_t* value_tensor_data = value_tensor_chunk_data + j * tensor_nbytes_;
      dst_buffers.push_back(value_tensor_data);
      src_buffers.push_back(value_tensor.data);
    }
  }

  vineyard::memory::concurrent_memcpy_n(dst_buffers, src_buffers,
                                        tensor_nbytes_);

  // write all tokens
  buffer += chunk_size_ * layer_ * tensor_nbytes_ * 2;
  std::memcpy(buffer, all_tokens_.data(), total_tokens_ * sizeof(int));

  is_ready_ = true;
  cv_.notify_all();

  if (VLOG_IS_ON(200)) {
    PrintKVCacheChunk();
  }

  return Status::OK();
}

std::shared_ptr<Object> KVCacheChunkBuilder::Seal() {
  VINEYARD_ASSERT(buffer_ == nullptr);

  if (remote_buffer_writer_ == nullptr) {
    return nullptr;
  }

  auto chunk = std::make_shared<KVCacheChunk>();

  // 1. seal the buffer
  ObjectMeta blob_meta;
  Status status =
      rpc_client_.CreateRemoteBlob(remote_buffer_writer_, blob_meta);
  if (!status.ok()) {
    VLOG(100) << "Failed to CreateRemoteBlob, error=" << status.ToString();
    return nullptr;
  }

  size_t nbytes = remote_buffer_writer_->size();
  chunk->meta_.AddMember(KVCacheChunk::kFieldNameBuffer, blob_meta);

  // 2. store the member field to meta
  chunk->meta_.AddKeyValue(KVCacheChunk::kFieldNameTotalTokens, total_tokens_);
  chunk->meta_.AddKeyValue(KVCacheChunk::kFieldNameChunkSize, chunk_size_);
  chunk->meta_.AddKeyValue(KVCacheChunk::kFieldNameTensorNBytes,
                           tensor_nbytes_);
  chunk->meta_.AddKeyValue(KVCacheChunk::kFieldNameLayer, layer_);
  chunk->meta_.AddKeyValue(KVCacheChunk::kFieldNameAccessTime, access_time_);
  chunk->meta_.AddKeyValue(KVCacheChunk::kFieldNameMd5, md5_);
  chunk->meta_.AddKeyValue(KVCacheChunk::kFieldNameNS, ns_);
  chunk->meta_.SetNBytes(nbytes);

  // 3. set the object type to meta
  chunk->meta_.SetTypeName(type_name<KVCacheChunk>());

  if (!rpc_client_.CreateMetaData(chunk->meta_, chunk->id_).ok()) {
    return nullptr;
  }

  return chunk;
}

void KVCacheChunkBuilder::PrintKVCacheChunk() {
  LOG(INFO) << "builder:" << this;

  uint8_t* buffer = nullptr;
  if (remote_buffer_writer_ != nullptr) {
    buffer = reinterpret_cast<uint8_t*>(remote_buffer_writer_->data());
  } else if (buffer_ != nullptr) {
    buffer = const_cast<uint8_t*>(buffer_->data());
  }

  if (buffer == nullptr) {
    LOG(INFO) << ">failed chunk";
  }

  if (total_tokens_ > chunk_size_) {
    std::string prefix_tokens = "";
    for (size_t i = 0; i < total_tokens_ - chunk_size_; i++) {
      prefix_tokens += std::to_string(all_tokens_[i]) + " ";
    }
    LOG(INFO) << ">prefix tokens:" << prefix_tokens;
  } else {
    LOG(INFO) << ">prefix tokens: N/A";
  }

  for (int i = 0; i < chunk_size_; i++) {
    LOG(INFO) << ">index:" << i;
    LOG(INFO) << ">token:" << all_tokens_[total_tokens_ - chunk_size_ + i];

    uint8_t* key_tensor_chunk_data = buffer + i * layer_ * tensor_nbytes_;
    uint8_t* value_tensor_chunk_data =
        key_tensor_chunk_data + chunk_size_ * layer_ * tensor_nbytes_;

    for (int curr = 0; curr < layer_; curr++) {
      LOG(INFO) << ">layer:" << curr;
      uint8_t* key_tensor_data = key_tensor_chunk_data + curr * tensor_nbytes_;
      uint8_t* value_tensor_data =
          value_tensor_chunk_data + curr * tensor_nbytes_;

      // print the first tensor_nbytes bytes
      std::string key_tensor = "";
      std::string value_tensor = "";
      for (int j = 0; j < tensor_nbytes_; j++) {
        key_tensor += std::to_string(key_tensor_data[j]) + " ";
        value_tensor += std::to_string(value_tensor_data[j]) + " ";
      }
      LOG(INFO) << ">>key_tensor:" << key_tensor;
      LOG(INFO) << ">>value_tensor:" << value_tensor;
    }
  }

  static const auto get_ts =
      [](std::chrono::duration<int64_t, std::nano> time) {
        auto duration_since_epoch =
            std::chrono::duration_cast<std::chrono::system_clock::duration>(
                time);
        std::chrono::time_point<std::chrono::system_clock> timestamp =
            std::chrono::system_clock::time_point(duration_since_epoch);
        time_t t = std::chrono::system_clock::to_time_t(timestamp);

        std::tm tm;
        localtime_r(&t, &tm);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
        return oss.str();
      };

  LOG(INFO) << ">global_access_time:"
            << get_ts(std::chrono::nanoseconds(g_access_time_));
  LOG(INFO) << ">access_time:"
            << get_ts(std::chrono::nanoseconds(access_time_));

  LOG(INFO) << "==========================";
}

}  // namespace vineyard
