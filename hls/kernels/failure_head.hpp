// Integer failure head (T39.5): the arithmetic of python/yolo/quant/int_failure.py, for the
// C-sim stage (and later the side kernel of T29.4, T39.6).
//
//   acc_i = Σ_j qW1[i][j]·(p[j] − zp[j]) + qb1[i]        int32 (|acc| < 2³¹ for d ≤ 1,536)
//   h_i   = clip((acc_i·M0_i + 2ⁿ⁻¹) >> n, 0, 127)       requantization as §9.3, then ReLU
//   score = sign·(Σ_i qW2[i]·h_i + qb2)                   int32 logit, high = alarm
//
// p: the integer pooled features (per channel, rounded sum >> ⌈log2 HW⌉, then max).
#pragma once

#include <cstdint>

namespace failure {

struct Head {
  int d, h, n, sign;
  int32_t qb2;
  const int32_t* zp;   // d (int8 range in hardware)
  const int8_t* qW1;   // h × d
  const int32_t* qb1;  // h
  const int32_t* M0;   // h
  const int8_t* qW2;   // h
};

inline int64_t rshift_round(int64_t x, int n) { return (x + (int64_t(1) << (n - 1))) >> n; }

inline int64_t score(const Head& m, const int32_t* p) {
  int64_t out = m.qb2;
  for (int i = 0; i < m.h; ++i) {
    int64_t acc = m.qb1[i];
    for (int j = 0; j < m.d; ++j) acc += int64_t(m.qW1[int64_t(i) * m.d + j]) * (p[j] - m.zp[j]);
    int64_t y = rshift_round(acc * m.M0[i], m.n);
    y = y < 0 ? 0 : (y > 127 ? 127 : y);
    out += int64_t(m.qW2[i]) * y;
  }
  return m.sign * out;
}

}  // namespace failure
