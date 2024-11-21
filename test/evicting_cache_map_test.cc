/*
 * Copyright 2014-present Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <set>

#include "common/util/evicting_cache_map.h"
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

void SanityTest() {
  EvictingCacheMap<int, int> map(0);

  CHECK_EQ(0, map.size());
  CHECK(map.empty());
  CHECK(!map.exists(1));
  map.set(1, 1);
  CHECK_EQ(1, map.size());
  CHECK(!map.empty());
  CHECK_EQ(1, map.get(1));
  CHECK(map.exists(1));
  map.set(1, 2);
  CHECK_EQ(1, map.size());
  CHECK(!map.empty());
  CHECK_EQ(2, map.get(1));
  CHECK(map.exists(1));
  map.erase(1);
  CHECK_EQ(0, map.size());
  CHECK(map.empty());
  CHECK(!map.exists(1));

  CHECK_EQ(0, map.size());
  CHECK(map.empty());
  CHECK(!map.exists(1));
  map.set(1, 1);
  CHECK_EQ(1, map.size());
  CHECK(!map.empty());
  CHECK_EQ(1, map.get(1));
  CHECK(map.exists(1));
  map.set(1, 2);
  CHECK_EQ(1, map.size());
  CHECK(!map.empty());
  CHECK_EQ(2, map.get(1));
  CHECK(map.exists(1));

  CHECK(!map.exists(2));
  map.set(2, 1);
  CHECK(map.exists(2));
  CHECK_EQ(2, map.size());
  CHECK(!map.empty());
  CHECK_EQ(1, map.get(2));
  map.set(2, 2);
  CHECK_EQ(2, map.size());
  CHECK(!map.empty());
  CHECK_EQ(2, map.get(2));
  CHECK(map.exists(2));
  map.erase(2);
  CHECK_EQ(1, map.size());
  CHECK(!map.empty());
  CHECK(!map.exists(2));
  map.erase(1);
  CHECK_EQ(0, map.size());
  CHECK(map.empty());
  CHECK(!map.exists(1));
}

void PruneTest() {
  EvictingCacheMap<int, int> map(0);
  CHECK_EQ(0, map.size());
  CHECK(map.empty());
  for (int i = 0; i < 100; i++) {
    CHECK(!map.exists(i));
  }

  for (int i = 0; i < 100; i++) {
    map.set(i, i);
    CHECK_EQ(i + 1, map.size());
    CHECK(!map.empty());
    CHECK(map.exists(i));
    CHECK_EQ(i, map.get(i));
  }

  map.prune(1000000);
  CHECK_EQ(0, map.size());
  CHECK(map.empty());
  for (int i = 0; i < 100; i++) {
    CHECK(!map.exists(i));
  }

  for (int i = 0; i < 100; i++) {
    map.set(i, i);
    CHECK_EQ(i + 1, map.size());
    CHECK(!map.empty());
    CHECK(map.exists(i));
    CHECK_EQ(i, map.get(i));
  }

  map.prune(100);
  CHECK_EQ(0, map.size());
  CHECK(map.empty());
  for (int i = 0; i < 100; i++) {
    CHECK(!map.exists(i));
  }

  for (int i = 0; i < 100; i++) {
    map.set(i, i);
    CHECK_EQ(i + 1, map.size());
    CHECK(!map.empty());
    CHECK(map.exists(i));
    CHECK_EQ(i, map.get(i));
  }

  map.prune(99);
  CHECK_EQ(1, map.size());
  CHECK(!map.empty());
  for (int i = 0; i < 99; i++) {
    CHECK(!map.exists(i));
  }
  CHECK(map.exists(99));
  CHECK_EQ(99, map.get(99));

  map.prune(100);
  CHECK_EQ(0, map.size());
  CHECK(map.empty());
  for (int i = 0; i < 100; i++) {
    CHECK(!map.exists(i));
  }

  for (int i = 0; i < 100; i++) {
    map.set(i, i);
    CHECK_EQ(i + 1, map.size());
    CHECK(!map.empty());
    CHECK(map.exists(i));
    CHECK_EQ(i, map.get(i));
  }

  map.prune(90);
  CHECK_EQ(10, map.size());
  CHECK(!map.empty());
  for (int i = 0; i < 90; i++) {
    CHECK(!map.exists(i));
  }
  for (int i = 90; i < 100; i++) {
    CHECK(map.exists(i));
    CHECK_EQ(i, map.get(i));
  }
}

void PruneHookTest() {
  EvictingCacheMap<int, int> map(0);
  CHECK_EQ(0, map.size());
  CHECK(map.empty());
  for (int i = 0; i < 100; i++) {
    CHECK(!map.exists(i));
  }

  int sum = 0;
  auto pruneCb = [&](auto&& pairs) {
    for (auto&& [k, v] : pairs) {
      CHECK_EQ(k, v);
      sum += k;
    }
  };

  map.setPruneHook(pruneCb);

  for (int i = 0; i < 100; i++) {
    map.set(i, i);
    CHECK_EQ(i + 1, map.size());
    CHECK(!map.empty());
    CHECK(map.exists(i));
    CHECK_EQ(i, map.get(i));
  }

  map.prune(1000000);
  CHECK_EQ(0, map.size());
  CHECK(map.empty());
  for (int i = 0; i < 100; i++) {
    CHECK(!map.exists(i));
  }
  CHECK_EQ((99 * 100) / 2, sum);
  sum = 0;

  for (int i = 0; i < 100; i++) {
    map.set(i, i);
    CHECK_EQ(i + 1, map.size());
    CHECK(!map.empty());
    CHECK(map.exists(i));
    CHECK_EQ(i, map.get(i));
  }

  map.prune(100);
  CHECK_EQ(0, map.size());
  CHECK(map.empty());
  for (int i = 0; i < 100; i++) {
    CHECK(!map.exists(i));
  }
  CHECK_EQ((99 * 100) / 2, sum);
  sum = 0;

  for (int i = 0; i < 100; i++) {
    map.set(i, i);
    CHECK_EQ(i + 1, map.size());
    CHECK(!map.empty());
    CHECK(map.exists(i));
    CHECK_EQ(i, map.get(i));
  }

  map.prune(99);
  CHECK_EQ(1, map.size());
  CHECK(!map.empty());
  for (int i = 0; i < 99; i++) {
    CHECK(!map.exists(i));
  }
  CHECK(map.exists(99));
  CHECK_EQ(99, map.get(99));

  CHECK_EQ((98 * 99) / 2, sum);
  sum = 0;

  map.prune(100);
  CHECK_EQ(0, map.size());
  CHECK(map.empty());
  for (int i = 0; i < 100; i++) {
    CHECK(!map.exists(i));
  }

  CHECK_EQ(99, sum);
  sum = 0;

  for (int i = 0; i < 100; i++) {
    map.set(i, i);
    CHECK_EQ(i + 1, map.size());
    CHECK(!map.empty());
    CHECK(map.exists(i));
    CHECK_EQ(i, map.get(i));
  }

  map.prune(90);
  CHECK_EQ(10, map.size());
  CHECK(!map.empty());
  for (int i = 0; i < 90; i++) {
    CHECK(!map.exists(i));
  }
  for (int i = 90; i < 100; i++) {
    CHECK(map.exists(i));
    CHECK_EQ(i, map.get(i));
  }
  CHECK_EQ((89 * 90) / 2, sum);
  sum = 0;
}

void SetMaxSize() {
  EvictingCacheMap<int, int> map(100, 20);
  for (int i = 0; i < 90; i++) {
    map.set(i, i);
    CHECK(map.exists(i));
  }

  CHECK_EQ(90, map.size());
  map.setMaxSize(50);
  CHECK_EQ(map.size(), 50);

  for (int i = 0; i < 90; i++) {
    map.set(i, i);
    CHECK(map.exists(i));
  }
  CHECK_EQ(40, map.size());
  map.setMaxSize(0);
  CHECK_EQ(40, map.size());
  map.setMaxSize(10);
  CHECK_EQ(10, map.size());
}

void SetClearSize() {
  EvictingCacheMap<int, int> map(100, 20);
  for (int i = 0; i < 90; i++) {
    map.set(i, i);
    CHECK(map.exists(i));
  }

  CHECK_EQ(90, map.size());
  map.setClearSize(40);
  map.setMaxSize(50);
  CHECK_EQ(map.size(), 50);

  for (int i = 0; i < 90; i++) {
    map.set(i, i);
    CHECK(map.exists(i));
  }
  CHECK_EQ(20, map.size());
  map.setMaxSize(0);
  CHECK_EQ(20, map.size());
  map.setMaxSize(10);
  CHECK_EQ(0, map.size());
}

void DestructorInvocationTest() {
  struct SumInt {
    SumInt(int val_, int* ref_) : val(val_), ref(ref_) {}
    ~SumInt() { *ref += val; }

    SumInt(SumInt const&) = delete;
    SumInt& operator=(SumInt const&) = delete;

    SumInt(SumInt&& other) : val(std::exchange(other.val, 0)), ref(other.ref) {}
    SumInt& operator=(SumInt&& other) {
      std::swap(val, other.val);
      std::swap(ref, other.ref);
      return *this;
    }

    int val;
    int* ref;
  };

  int sum;
  EvictingCacheMap<int, SumInt> map(0);

  CHECK_EQ(0, map.size());
  CHECK(map.empty());
  for (int i = 0; i < 100; i++) {
    CHECK(!map.exists(i));
  }

  for (int i = 0; i < 100; i++) {
    map.set(i, SumInt(i, &sum));
    CHECK_EQ(i + 1, map.size());
    CHECK(!map.empty());
    CHECK(map.exists(i));
    CHECK_EQ(i, map.get(i).val);
  }

  sum = 0;
  map.prune(1000000);
  CHECK_EQ(0, map.size());
  CHECK(map.empty());
  for (int i = 0; i < 100; i++) {
    CHECK(!map.exists(i));
  }
  CHECK_EQ((99 * 100) / 2, sum);

  for (int i = 0; i < 100; i++) {
    map.set(i, SumInt(i, &sum));
    CHECK_EQ(i + 1, map.size());
    CHECK(!map.empty());
    CHECK(map.exists(i));
    CHECK_EQ(i, map.get(i).val);
  }

  sum = 0;
  map.prune(100);
  CHECK_EQ(0, map.size());
  CHECK(map.empty());
  for (int i = 0; i < 100; i++) {
    CHECK(!map.exists(i));
  }
  CHECK_EQ((99 * 100) / 2, sum);

  for (int i = 0; i < 100; i++) {
    map.set(i, SumInt(i, &sum));
    CHECK_EQ(i + 1, map.size());
    CHECK(!map.empty());
    CHECK(map.exists(i));
    CHECK_EQ(i, map.get(i).val);
  }

  sum = 0;
  map.prune(99);
  CHECK_EQ(1, map.size());
  CHECK(!map.empty());
  for (int i = 0; i < 99; i++) {
    CHECK(!map.exists(i));
  }
  CHECK(map.exists(99));
  CHECK_EQ(99, map.get(99).val);

  CHECK_EQ((98 * 99) / 2, sum);

  sum = 0;
  map.prune(100);
  CHECK_EQ(0, map.size());
  CHECK(map.empty());
  for (int i = 0; i < 100; i++) {
    CHECK(!map.exists(i));
  }

  CHECK_EQ(99, sum);
  for (int i = 0; i < 100; i++) {
    map.set(i, SumInt(i, &sum));
    CHECK_EQ(i + 1, map.size());
    CHECK(!map.empty());
    CHECK(map.exists(i));
    CHECK_EQ(i, map.get(i).val);
  }

  sum = 0;
  map.prune(90);
  CHECK_EQ(10, map.size());
  CHECK(!map.empty());
  for (int i = 0; i < 90; i++) {
    CHECK(!map.exists(i));
  }
  for (int i = 90; i < 100; i++) {
    CHECK(map.exists(i));
    CHECK_EQ(i, map.get(i).val);
  }
  CHECK_EQ((89 * 90) / 2, sum);
  sum = 0;
  for (int i = 0; i < 90; i++) {
    auto pair = map.insert(i, SumInt(i + 1, &sum));
    CHECK_EQ(i + 1, pair.first->second.val);
    CHECK(pair.second);
    CHECK(map.exists(i));
  }
  CHECK_EQ(0, sum);
  for (int i = 90; i < 100; i++) {
    auto pair = map.insert(i, SumInt(i + 1, &sum));
    CHECK_EQ(i, pair.first->second.val);
    CHECK(!pair.second);
    CHECK(map.exists(i));
  }
  CHECK_EQ((10 * 191) / 2, sum);
  sum = 0;
  map.prune(100);
  CHECK_EQ((90 * 91) / 2 + (10 * 189) / 2, sum);

  sum = 0;
  map.set(3, SumInt(3, &sum));
  map.set(2, SumInt(2, &sum));
  map.set(1, SumInt(1, &sum));
  CHECK_EQ(0, sum);
  CHECK_EQ(2, map.erase(map.find(1))->second.val);
  CHECK_EQ(1, sum);
  CHECK(map.end() == map.erase(map.findWithoutPromotion(3)));
  CHECK_EQ(4, sum);
  map.prune(1);
  CHECK_EQ(6, sum);
}

void LruSanityTest() {
  EvictingCacheMap<int, int> map(10);
  CHECK_EQ(0, map.size());
  CHECK(map.empty());
  for (int i = 0; i < 100; i++) {
    CHECK(!map.exists(i));
  }

  for (int i = 0; i < 100; i++) {
    map.set(i, i);
    CHECK_GE(10, map.size());
    CHECK(!map.empty());
    CHECK(map.exists(i));
    CHECK_EQ(i, map.get(i));
  }

  CHECK_EQ(10, map.size());
  CHECK(!map.empty());
  for (int i = 0; i < 90; i++) {
    CHECK(!map.exists(i));
  }
  for (int i = 90; i < 100; i++) {
    CHECK(map.exists(i));
  }
}

void LruPromotionTest() {
  EvictingCacheMap<int, int> map(10);
  CHECK_EQ(0, map.size());
  CHECK(map.empty());
  for (int i = 0; i < 100; i++) {
    CHECK(!map.exists(i));
  }

  for (int i = 0; i < 100; i++) {
    map.set(i, i);
    CHECK_GE(10, map.size());
    CHECK(!map.empty());
    CHECK(map.exists(i));
    CHECK_EQ(i, map.get(i));
    for (int j = 0; j < std::min(i + 1, 9); j++) {
      CHECK(map.exists(j));
      CHECK_EQ(j, map.get(j));
    }
  }

  CHECK_EQ(10, map.size());
  CHECK(!map.empty());
  for (int i = 0; i < 9; i++) {
    CHECK(map.exists(i));
  }
  CHECK(map.exists(99));
  for (int i = 10; i < 99; i++) {
    CHECK(!map.exists(i));
  }
}

void LruNoPromotionTest() {
  EvictingCacheMap<int, int> map(10);
  CHECK_EQ(0, map.size());
  CHECK(map.empty());
  for (int i = 0; i < 100; i++) {
    CHECK(!map.exists(i));
  }

  for (int i = 0; i < 100; i++) {
    map.set(i, i);
    CHECK_GE(10, map.size());
    CHECK(!map.empty());
    CHECK(map.exists(i));
    CHECK_EQ(i, map.get(i));
    for (int j = 0; j < std::min(i + 1, 9); j++) {
      if (map.exists(j)) {
        CHECK_EQ(j, map.getWithoutPromotion(j));
      }
    }
  }

  CHECK_EQ(10, map.size());
  CHECK(!map.empty());
  for (int i = 0; i < 90; i++) {
    CHECK(!map.exists(i));
  }
  for (int i = 90; i < 100; i++) {
    CHECK(map.exists(i));
  }
}

void IteratorSanityTest() {
  const int nItems = 1000;
  EvictingCacheMap<int, int> map(nItems);
  CHECK(map.begin() == map.end());
  for (int i = 0; i < nItems; i++) {
    CHECK(!map.exists(i));
    map.set(i, i * 2);
    CHECK(map.exists(i));
    CHECK_EQ(i * 2, map.get(i));
  }

  std::set<int> seen;
  for (auto& it : map) {
    CHECK_EQ(0, seen.count(it.first));
    seen.insert(it.first);
    CHECK_EQ(it.first * 2, it.second);
  }
  CHECK_EQ(nItems, seen.size());
}

void FindTest() {
  const int nItems = 1000;
  EvictingCacheMap<int, int> map(nItems);
  for (int i = 0; i < nItems; i++) {
    map.set(i * 2, i * 2);
    CHECK(map.exists(i * 2));
    CHECK_EQ(i * 2, map.get(i * 2));
  }
  for (int i = 0; i < nItems * 2; i++) {
    if (i % 2 == 0) {
      auto it = map.find(i);
      CHECK(it != map.end());
      CHECK_EQ(i, it->first);
      CHECK_EQ(i, it->second);
    } else {
      CHECK(map.find(i) == map.end());
    }
  }
  for (int i = nItems * 2 - 1; i >= 0; i--) {
    if (i % 2 == 0) {
      auto it = map.find(i);
      CHECK(it != map.end());
      CHECK_EQ(i, it->first);
      CHECK_EQ(i, it->second);
    } else {
      CHECK(map.find(i) == map.end());
    }
  }
  CHECK_EQ(0, map.begin()->first);
}

void FindWithoutPromotionTest() {
  const int nItems = 1000;
  EvictingCacheMap<int, int> map(nItems);
  for (int i = 0; i < nItems; i++) {
    map.set(i * 2, i * 2);
    CHECK(map.exists(i * 2));
    CHECK_EQ(i * 2, map.get(i * 2));
  }
  for (int i = nItems * 2 - 1; i >= 0; i--) {
    if (i % 2 == 0) {
      auto it = map.findWithoutPromotion(i);
      CHECK(it != map.end());
      CHECK_EQ(i, it->first);
      CHECK_EQ(i, it->second);
    } else {
      CHECK(map.findWithoutPromotion(i) == map.end());
    }
  }
  CHECK_EQ((nItems - 1) * 2, map.begin()->first);
}

void IteratorOrderingTest() {
  const int nItems = 1000;
  EvictingCacheMap<int, int> map(nItems);
  for (int i = 0; i < nItems; i++) {
    map.set(i, i);
    CHECK(map.exists(i));
    CHECK_EQ(i, map.get(i));
  }

  int expected = nItems - 1;
  for (auto it = map.begin(); it != map.end(); ++it) {
    CHECK_EQ(expected, it->first);
    expected--;
  }

  expected = 0;
  for (auto it = map.rbegin(); it != map.rend(); ++it) {
    CHECK_EQ(expected, it->first);
    expected++;
  }

  {
    auto it = map.end();
    expected = 0;
    CHECK(it != map.begin());
    do {
      --it;
      CHECK_EQ(expected, it->first);
      expected++;
    } while (it != map.begin());
    CHECK_EQ(nItems, expected);
  }

  {
    auto it = map.rend();
    expected = nItems - 1;
    do {
      --it;
      CHECK_EQ(expected, it->first);
      expected--;
    } while (it != map.rbegin());
    CHECK_EQ(-1, expected);
  }
}

void MoveTest() {
  const int nItems = 1000;
  EvictingCacheMap<int, int> map(nItems);
  for (int i = 0; i < nItems; i++) {
    map.set(i, i);
    CHECK(map.exists(i));
    CHECK_EQ(i, map.get(i));
  }

  EvictingCacheMap<int, int> map2 = std::move(map);
  CHECK(map.empty());
  for (int i = 0; i < nItems; i++) {
    CHECK(map2.exists(i));
    CHECK_EQ(i, map2.get(i));
  }
}

void CustomKeyEqual() {
  const int nItems = 100;
  struct Eq {
    bool operator()(const int& a, const int& b) const {
      return (a % mod) == (b % mod);
    }
    int mod;
  };
  struct Hash {
    size_t operator()(const int& a) const { return std::hash<int>()(a % mod); }
    int mod;
  };
  EvictingCacheMap<int, int, Hash, Eq> map(nItems, 1 /* clearSize */,
                                           Hash{nItems}, Eq{nItems});
  for (int i = 0; i < nItems; i++) {
    map.set(i, i);
    CHECK(map.exists(i));
    CHECK_EQ(i, map.get(i));
    CHECK(map.exists(i + nItems));
    CHECK_EQ(i, map.get(i + nItems));
  }
}

void IteratorConversion() {
  using type = EvictingCacheMap<int, int>;
  using i = type::iterator;
  using ci = type::const_iterator;
  using ri = type::reverse_iterator;
  using cri = type::const_reverse_iterator;

  CHECK((std::is_convertible<i, i>::value));
  CHECK((std::is_convertible<i, ci>::value));
  CHECK(!(std::is_convertible<ci, i>::value));
  CHECK((std::is_convertible<ci, ci>::value));

  CHECK((std::is_convertible<ri, ri>::value));
  CHECK((std::is_convertible<ri, cri>::value));
  CHECK(!(std::is_convertible<cri, ri>::value));
  CHECK((std::is_convertible<cri, cri>::value));
}

int main(int argc, char** argv) {
  SanityTest();
  PruneTest();
  PruneHookTest();
  SetMaxSize();
  SetClearSize();
  DestructorInvocationTest();
  LruSanityTest();
  LruPromotionTest();
  LruNoPromotionTest();
  IteratorSanityTest();
  FindTest();
  FindWithoutPromotionTest();
  IteratorOrderingTest();
  MoveTest();
  CustomKeyEqual();
  IteratorConversion();

  LOG(INFO) << "All tests passed";
  return 0;
}
