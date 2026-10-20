// T28.1: the golden output stage against an independent specification in 128-bit integers,
// on every rounding boundary of every shift and on the int32 extremes (the Python == golden ==
// HLS sweep over the exported layers is python/tests/test_exhaustive_ops.py).
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <random>

#include "golden/conv.hpp"

namespace {

// ⌊(acc·M0 + 2ⁿ⁻¹) / 2ⁿ⌋ with a mathematical floor (no shift, no int64).
__int128 floor_div(__int128 a, __int128 b) {
  __int128 q = a / b;
  if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
  return q;
}

int32_t spec_requant(int32_t acc, int32_t m0, int n) {
  return int32_t(floor_div(__int128(acc) * m0 + (__int128(1) << (n - 1)), __int128(1) << n));
}

int32_t spec_leaky(int32_t y) { return y > 0 ? y : int32_t(floor_div(__int128(13) * y + 64, 128)); }

}  // namespace

TEST_CASE("requantize, leaky, clip_q == 128-bit spec on the rounding boundaries") {
  std::mt19937 rng(281);
  long long checked = 0;
  for (int n = 1; n <= 31; ++n) {
    for (int it = 0; it < 200; ++it) {
      const int32_t m0 = it == 0 ? 0x7fffffff : it == 1 ? 1 : int32_t(rng() & 0x7fffffff);
      for (int64_t k = -1400; k <= 1400; k += (it < 2 ? 1 : 7)) {
        // first accumulator with requant ≥ k, and its neighbours
        const __int128 num = (__int128(k) << n) - (__int128(1) << (n - 1));
        const __int128 b = m0 ? -floor_div(-num, m0) : 0;
        for (int dlt = -1; dlt <= 1; ++dlt) {
          __int128 v = b + dlt;
          if (v < INT32_MIN || v > INT32_MAX) continue;
          const int32_t acc = int32_t(v);
          REQUIRE(golden::requantize(acc, m0, n) == spec_requant(acc, m0, n));
          const int32_t y = spec_requant(acc, m0, n);
          REQUIRE(golden::leaky_int(y) == spec_leaky(y));
          for (int qmax : {7, 127}) {
            const int32_t s = spec_leaky(y);
            REQUIRE(int(golden::clip_q(s, qmax)) == (s > qmax ? qmax : s < -qmax ? -qmax : s));
          }
          ++checked;
        }
      }
      for (int32_t acc : {INT32_MIN, INT32_MIN + 1, -1, 0, 1, INT32_MAX - 1, INT32_MAX}) {
        REQUIRE(golden::requantize(acc, m0, n) == spec_requant(acc, m0, n));
        ++checked;
      }
    }
  }
  CHECK(checked > 1000000);
}
