// Fault hook of the C-sim (T25.3, T25.7): one bit flip in an on-chip buffer of `yolo_conv`,
// armed by the testbench (hls/tb/tb_fault.cpp) before a call, fired once. Absent from the
// synthesis (`__SYNTHESIS__`): no logic, no cost.
//
// Buffers and indices (the arrays of conv_pe.cpp and output_stage.hpp):
//   FB_IN  in_buf[a][b][c]      int8 input tile, before the compute of input-channel tile `ti`
//   FB_W   w_buf[a][b][c][d]    int8 weight tile, same moment
//   FB_OUT out_buf[a][b][c]     int32 accumulators of the tile, after the last `ti`
//   FB_Q   q[a][b][c]           int8 requantized tile read by the fused max-pool
//                               (a = output channel in the tile)
//   FB_CS  cs_buf[b][c]         ABFT checksum lane of the (row, col) tile
// `tile`: index of the output tile in the call (order row → col → to, as `conv_layer`).
#pragma once

#include <cstdint>

namespace accel {

enum FaultBuf { FB_NONE = -1, FB_IN = 0, FB_W = 1, FB_OUT = 2, FB_Q = 3, FB_CS = 4 };

#ifndef __SYNTHESIS__
struct HlsFault {
  int buf = FB_NONE;
  int tile = 0, ti = 0;
  int a = 0, b = 0, c = 0, d = 0;
  int bit = 0;
  int fired = 0;
  bool armed(int which, int t, int k = 0) const {
    return buf == which && !fired && tile == t && (which > FB_W || ti == k);
  }
};
extern HlsFault hls_fault;

template <class T>
inline void flip8(T& v, int bit) {
  v = T(int8_t(uint8_t(int8_t(v)) ^ uint8_t(1u << bit)));
}
template <class T>
inline void flip32(T& v, int bit) {
  v = T(int32_t(uint32_t(int32_t(v)) ^ (1u << bit)));
}
#endif

}  // namespace accel
