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

#include <unistd.h>
#include <iostream>
#include <random>
#include <vector>

#include "client/client.h"
#include "client/ds/object_meta.h"
#include "client/rpc_client.h"
#include "common/util/logging.h"
#include "llm-cache/ds/config.h"
#include "llm-cache/ds/kv_cache_manager.h"

using namespace vineyard;  // NOLINT(build/namespaces)

size_t nr_rounds = 3;
int tensorNBytes = 80;
int capacity = 20;
int layer = 3;
int chunk_size = 5;
std::string cache_prefix = "aibrix_test";
int global_ttl = 5;

AIBrixCacheConfig config;

std::vector<int> round_1_tokens = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                   12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                   23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                                   34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                                   45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                                   56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
                                   67, 68, 69, 70};  // 70 tokens
std::vector<int> round_2_tokens = {1,  2,  3,  4,  5,  7,  8, 9,
                                   10, 11, 12, 13, 14, 15, 51};  // 15 tokens
std::vector<int> round_3_tokens = {1,  2,  3,  9,  10,
                                   11, 12, 13, 14, 21};  // 10 tokens
std::vector<int> round_4_tokens = {1, 2, 3, 4, 5};       // 5 tokens

std::vector<std::vector<int>> tokens_list = {round_1_tokens, round_2_tokens,
                                             round_3_tokens, round_4_tokens};

std::shared_ptr<KVCacheManager> init(RPCClient& rpc_client, Client& client) {
  std::shared_ptr<KVCacheManager> kv_cache_manager;
  VINEYARD_CHECK_OK(
      KVCacheManager::Make(rpc_client, client, kv_cache_manager, config));
  return kv_cache_manager;
}

void print_current_tokens(const std::vector<int>& prefix, int next_token) {
  std::string tokens_str = "";
  for (size_t i = 0; i < prefix.size(); ++i) {
    tokens_str += std::to_string(prefix[i]) + " ";
  }
  tokens_str += std::to_string(next_token);
  LOG(INFO) << "Current tokens: " + tokens_str;
}

void print_kv_state(const std::vector<std::pair<LLMKV, LLMKV>>& kv_state) {
  VLOG(100) << "kv_state: ";
  for (size_t i = 0; i < kv_state.size(); ++i) {
    uint8_t* key_state_data =
        reinterpret_cast<uint8_t*>(kv_state[i].first.data);
    uint8_t* value_state_data =
        reinterpret_cast<uint8_t*>(kv_state[i].second.data);
    // print the first tensorNBytes bytes
    std::string key_state_str = "";
    std::string value_state_str = "";
    for (int j = 0; j < tensorNBytes; j++) {
      key_state_str += std::to_string(key_state_data[j]) + " ";
      value_state_str += std::to_string(value_state_data[j]) + " ";
    }
    VLOG(100) << "layer " << i << ":";
    VLOG(100) << "key_state: " << key_state_str;
    VLOG(100) << "value_state: " << value_state_str;
    VLOG(100) << "---------------------";
  }
}

// we do not consider the layer.
std::vector<std::pair<LLMKV, LLMKV>> generate_kv_state(int token) {
  std::vector<std::pair<LLMKV, LLMKV>> kv_state;
  for (int currentLayer = 0; currentLayer < layer; currentLayer++) {
    LLMKV key_state;
    LLMKV value_state;
    key_state.data = malloc(tensorNBytes);
    value_state.data = malloc(tensorNBytes);

    key_state.length = tensorNBytes;
    value_state.length = tensorNBytes;

    for (int i = 0; i < tensorNBytes; ++i) {
      (reinterpret_cast<uint8_t*>(key_state.data))[i] =
          (static_cast<uint8_t>(token)) + i + currentLayer;
      (reinterpret_cast<uint8_t*>(value_state.data))[i] =
          (static_cast<uint8_t>(token)) + i + currentLayer;
    }
    kv_state.emplace_back(key_state, value_state);
  }
  return kv_state;
}

void check_kv_state(const std::vector<std::pair<LLMKV, LLMKV>>& kv_state,
                    int& token) {
  VINEYARD_ASSERT(kv_state.size() == (size_t) layer);
  for (size_t index = 0; index < kv_state.size(); ++index) {
    VINEYARD_ASSERT(kv_state[index].first.length == (size_t) tensorNBytes);
    VINEYARD_ASSERT(kv_state[index].second.length == (size_t) tensorNBytes);
    for (int i = 0; i < tensorNBytes; ++i) {
      if ((reinterpret_cast<uint8_t*>(kv_state[index].first.data))[i] !=
          (static_cast<uint8_t>(token)) + i + index) {
        VLOG(100) << "token:" << token << " tensorNBytes" << tensorNBytes
                  << " layer:" << index;
        VLOG(100) << "key_state[" << i << "]: "
                  << (reinterpret_cast<uint8_t*>(kv_state[index].first.data))[i]
                  << ". But is should be "
                  << (static_cast<uint8_t>(token)) + i + index;
        throw std::runtime_error("key_state error!");
      }
      if (reinterpret_cast<uint8_t*>(kv_state[index].second.data)[i] !=
          (static_cast<uint8_t>(token)) + i + index) {
        VLOG(100) << "token:" << token << " tensorNBytes" << tensorNBytes
                  << " layer:" << index;
        VLOG(100) << "value_state[" << i << "]: "
                  << (reinterpret_cast<uint8_t*>(
                         kv_state[index].second.data))[i]
                  << ". But is should be "
                  << (static_cast<uint8_t>(token)) + i + index;
        throw std::runtime_error("value_state error!");
      }
    }
  }
}

void inference(std::shared_ptr<KVCacheManager>& kv_cache_manager,
               std::vector<int> tokens, bool block = false) {
  std::vector<int> inference_tokens;
  std::vector<std::vector<std::pair<LLMKV, LLMKV>>> kv_state;
  for (size_t i = 0; i < tokens.size(); ++i) {
    std::vector<std::pair<LLMKV, LLMKV>> current_kv_state =
        generate_kv_state(tokens[i]);
    print_kv_state(current_kv_state);
    kv_state.push_back(current_kv_state);
    inference_tokens.push_back(tokens[i]);
  }

  size_t updated = 0;
  Status result = kv_cache_manager->Update(inference_tokens, kv_state, updated);

  std::vector<std::vector<std::pair<LLMKV, LLMKV>>> kv_state_to_query;
  for (size_t i = 0; i < tokens.size(); ++i) {
    std::vector<std::pair<LLMKV, LLMKV>> current_kv_state =
        generate_kv_state(0);
    kv_state_to_query.push_back(current_kv_state);
  }
  size_t matched = 0;
  Status query_result =
      kv_cache_manager->Query(inference_tokens, kv_state_to_query, matched);
  if (!query_result.ok()) {
    LOG(INFO) << "Query failed!";
  }

  LOG(INFO) << "Match tokens:" << matched << ". Total tokens:" << tokens.size();
  for (size_t i = 0; i < matched; ++i) {
    check_kv_state(kv_state_to_query[i], tokens[i]);
  }
}

void inference(std::shared_ptr<KVCacheManager>& kv_cache_manager,
               std::vector<int> prefix, std::vector<int> tokens,
               bool block = false) {
  std::vector<int> inference_tokens;
  std::vector<std::pair<LLMKV, LLMKV>> kv_state_to_query;
  std::vector<std::vector<std::pair<LLMKV, LLMKV>>> kv_state_to_query_list;

  // Get tokens with batched query interface
  inference_tokens = std::vector(prefix.begin(), prefix.end());

  for (int i = 0; i < tokens.size(); i++) {
    kv_state_to_query.clear();
    for (int currentLayer = 0; currentLayer < layer; currentLayer++) {
      kv_state_to_query.emplace_back(LLMKV{nullptr, 0}, LLMKV{nullptr, 0});
    }
    kv_state_to_query_list.emplace_back(kv_state_to_query);
  }
  size_t matched = 0;
  Status result = kv_cache_manager->Query(inference_tokens, tokens,
                                          kv_state_to_query_list, matched);
  LOG(INFO) << "Find " << matched << " matched tokens from token lists.";
  for (size_t i = 0; i < matched; i++) {
    print_current_tokens(inference_tokens, tokens[i]);
    inference_tokens.emplace_back(tokens[i]);
    check_kv_state(kv_state_to_query_list[i], tokens[i]);
  }
}

void thread_func(std::string socket) {
  RPCClient dummy_rpc_client;
  Client client;
  VINEYARD_CHECK_OK(client.Connect(socket));
  std::shared_ptr<KVCacheManager> manager = init(dummy_rpc_client, client);

  for (size_t r = 0; r < nr_rounds; r++) {
    for (size_t i = 0; i < tokens_list.size(); i++) {
      LOG(INFO) << "Round " << i << " :";
      inference(manager, tokens_list[i]);
    }
    sleep(1);
  }

  sleep(5);

  for (size_t r = 0; r < nr_rounds; r++) {
    for (size_t i = 0; i < tokens_list.size(); i++) {
      LOG(INFO) << "Round " << i << " :";
      inference(manager, tokens_list[i]);
    }
    sleep(1);
  }

  sleep(5);

  for (size_t r = 0; r < nr_rounds; r++) {
    for (size_t i = 0; i < tokens_list.size(); i++) {
      size_t total_chunks = tokens_list[i].size() / chunk_size;
      size_t prefix_chunks = total_chunks / 2;
      std::vector<int> prefix(
          tokens_list[i].begin(),
          tokens_list[i].begin() + prefix_chunks * chunk_size);
      std::vector<int> tokens_list_rest(
          tokens_list[i].begin() + prefix_chunks * chunk_size,
          tokens_list[i].end());
      inference(manager, prefix, tokens_list_rest);
    }
    sleep(1);
  }

  LOG(INFO) << "inference end";

  // sleep a while to trigger local sync and global gc
  sleep(3 * global_ttl);

  manager->Close();
  client.Disconnect();
  if (dummy_rpc_client.Connected()) {
    dummy_rpc_client.Disconnect();
  }
}

void list_objects(const std::string& socket) {
  Client client;
  VINEYARD_CHECK_OK(client.Connect(socket));
  std::shared_ptr<InstanceStatus> status;
  VINEYARD_CHECK_OK(client.InstanceStatus(status));

  std::vector<ObjectMeta> metas;
  if (status->memory_usage != 0) {
    metas = client.ListObjectMeta(".*", true);
    LOG(INFO) << "Object:";
    for (size_t i = 0; i < metas.size(); i++) {
      LOG(INFO) << metas[i].ToString();
    }
  }

  client.Disconnect();
}

void clear_objects(const std::string& socket) {
  Client client;
  VINEYARD_CHECK_OK(client.Connect(socket));
  std::shared_ptr<InstanceStatus> status;
  VINEYARD_CHECK_OK(client.InstanceStatus(status));

  std::vector<ObjectMeta> metas;
  if (status->memory_usage != 0) {
    metas = client.ListObjectMeta(".*", true);
    for (size_t i = 0; i < metas.size(); i++) {
      LOG(INFO) << "Client " << client.instance_id() << " deletes obj "
                << ObjectIDToString(metas[i].GetId())
                << " instance_id=" << metas[i].GetInstanceId();
      client.DelData(metas[i].GetId());
    }
  }

  client.Disconnect();
}

void test(const std::vector<std::string>& sockets, bool enableGlobalGC) {
  config = AIBrixCacheConfig(tensorNBytes, capacity, layer, chunk_size,
                             cache_prefix, 1, enableGlobalGC, 3, global_ttl);

  std::vector<std::thread> threads;
  for (int i = 0; i < sockets.size(); i++) {
    threads.push_back(std::thread(thread_func, sockets[i]));
  }

  for (int i = 0; i < sockets.size(); i++) {
    threads[i].join();
    LOG(INFO) << "Thread:" << i << " exit.";
    list_objects(sockets[i]);
  }

  for (int i = 0; i < sockets.size(); i++) {
    clear_objects(sockets[i]);
  }

  size_t total_memory_usage = 0;
  for (size_t i = 0; i < sockets.size(); i++) {
    Client client;
    VINEYARD_CHECK_OK(client.Connect(sockets[i]));
    std::shared_ptr<InstanceStatus> status;
    VINEYARD_CHECK_OK(client.InstanceStatus(status));
    LOG(INFO) << "Client " << client.instance_id()
              << " memory usage:" << status->memory_usage;
    total_memory_usage += status->memory_usage;
    client.Disconnect();
  }
  LOG(INFO) << "Total memory usage:" << total_memory_usage;
}

int main(int argc, char** argv) {
  std::vector<std::string> sockets;
  if (argc < 2) {
    printf(
        "usage ./aibrix_kv_cache_test --client-num <client_num> "
        "--vineyard-ipc-sockets <ipc_socket_1> ... <ipc_socket_n> -d "
        "<tensorNBytes> -c <capacity> -l <layer> -b <blockSize>\n");
    return 1;
  }

  if (strcmp(argv[1], "--client-num") != 0) {
    return 1;
  }

  int client_num = std::stoi(argv[2]);

  for (int i = 3; i < argc; i++) {
    if (strcmp(argv[i], "-d") == 0) {
      tensorNBytes = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-c") == 0) {
      capacity = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-l") == 0) {
      layer = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-b") == 0) {
      chunk_size = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-s") == 0) {
      for (int j = i + 1; j < argc; j++) {
        if (strcmp(argv[j], "1") == 0) {
          tokens_list.push_back(round_1_tokens);
        } else if (strcmp(argv[j], "2") == 0) {
          tokens_list.push_back(round_2_tokens);
        } else if (strcmp(argv[j], "3") == 0) {
          tokens_list.push_back(round_3_tokens);
        } else if (strcmp(argv[j], "4") == 0) {
          tokens_list.push_back(round_4_tokens);
        } else {
          break;
        }
      }
    } else if (strcmp(argv[i], "--vineyard-ipc-sockets") == 0) {
      for (int j = 0; j < client_num; j++) {
        sockets.push_back(std::string(argv[i + j + 1]));
      }
    }
  }

  LOG(INFO) << "Test AIBrixKVCache with tensorNBytes: " << tensorNBytes
            << ", capacity: " << capacity << ", layer: " << layer
            << ", chunk_size: " << chunk_size
            << ", cache_prefix: " << cache_prefix << " and use " << client_num
            << " client.";

  test(sockets, /* global gc */ false);
  test(sockets, /* global gc */ true);

  LOG(INFO) << "Passed AIBrixKVCache tests...";
  return 0;
}
