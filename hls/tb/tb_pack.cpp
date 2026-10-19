// T23.2: packed product of the kernel (accel::mac2, ap_int 27 × 18 → 48 bits) == golden::mac2
// and exact unpacking, on every int8 triple (w_a, w_b, x).
#include <cstdio>

#include "accel.hpp"

int main() {
  long long bad = 0;
  for (int wa = -128; wa < 128; ++wa)
    for (int wb = -128; wb < 128; ++wb)
      for (int x = -128; x < 128; ++x) {
        const accel::pack_t p = accel::mac2(accel::w_t(wa), accel::w_t(wb), accel::act_t(x));
        int32_t hi, lo;
        golden::unpack2(int64_t(p), hi, lo);
        bad += int64_t(p) != golden::mac2(int8_t(wa), int8_t(wb), int8_t(x)) || hi != wa * x ||
               lo != wb * x;
      }
  std::printf("mac2: %lld mismatches over 2^24 triples\n", bad);
  return bad != 0;
}
