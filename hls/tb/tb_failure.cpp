// Failure head at the C-sim stage (T39.5): reads the blob of `IntFailureHead.blob` (int32
// little-endian: d, h, n, qb2, count, zp, qW1, qb1, M0, qW2, inputs) and prints one score per
// input vector, for python/tests/test_failure.py and tools/failure_prediction.py stages.
//
//   tb_failure <blob> [sign]
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "failure_head.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: tb_failure <blob> [sign]\n");
    return 2;
  }
  std::ifstream f(argv[1], std::ios::binary);
  std::vector<int32_t> v;
  int32_t x;
  while (f.read(reinterpret_cast<char*>(&x), sizeof x)) v.push_back(x);
  if (v.size() < 5) return 1;
  const int d = v[0], h = v[1], n = v[2], count = v[4];
  const size_t need = 5 + size_t(d) + size_t(h) * d + 3 * size_t(h) + size_t(count) * d;
  if (v.size() != need) {
    std::fprintf(stderr, "blob: %zu words, expected %zu\n", v.size(), need);
    return 1;
  }
  const int32_t* zp = v.data() + 5;
  std::vector<int8_t> qW1(v.begin() + 5 + d, v.begin() + 5 + d + size_t(h) * d);
  const int32_t* qb1 = zp + d + size_t(h) * d;
  const int32_t* M0 = qb1 + h;
  std::vector<int8_t> qW2(M0 + h, M0 + 2 * h);
  const int32_t* p = M0 + 2 * h;
  failure::Head m{d, h, n, argc > 2 ? std::atoi(argv[2]) : 1, v[3], zp, qW1.data(), qb1,
                  M0, qW2.data()};
  for (int k = 0; k < count; ++k)
    std::printf("%lld\n", (long long)failure::score(m, p + size_t(k) * d));
  return 0;
}
