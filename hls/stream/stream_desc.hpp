// Registres de `yolo_stream` (T9.4.2, T10.9) : C++ simple, partagé par le noyau, le testbench
// et le driver ARM (backend uio compris, sans le reste de hls/stream).
#pragma once

#include <cstdint>

namespace stream {

constexpr int N_STAGES = 9;
constexpr int STAGE_LAYER[N_STAGES] = {0, 2, 4, 6, 8, 10, 12, 13, 14};

constexpr int IN_VALUES = 3 * 416 * 416;   // entrée (3, 416, 416), HWC
constexpr int OUT_VALUES = 125 * 13 * 13;  // tête (125, 13, 13), HWC

// Par étage : offsets dans les blobs (poids en octets, biais et M0 en indices int32), décalage
// et saturation. Avec STREAM_ROM, w_off ne sert qu'aux étages FRAME (poids en DDR).
struct StreamDesc {
  int32_t w_off[N_STAGES];
  int32_t b_off[N_STAGES];
  int32_t m0_off[N_STAGES];
  int32_t shift[N_STAGES];
  int32_t qmax[N_STAGES];
};

constexpr int S_WORDS = 5 * N_STAGES;
static_assert(sizeof(StreamDesc) == 4 * S_WORDS, "StreamDesc : 45 × int32");

}  // namespace stream
