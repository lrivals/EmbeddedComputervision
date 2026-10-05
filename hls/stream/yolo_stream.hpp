// Tiny-YOLOv2 en architecture streaming (T9.4.2) : 9 étages conv chaînés par des flux sous
// DATAFLOW. Repliements PE × SIMD : plan de tools/stream_model.py pour la KV260 (poids 8
// bits) ; L12 et L13 gardent leur carte d'entrée et lisent leurs poids en DDR (hybride).
#pragma once

#include <cstdint>
#include <vector>

#include "conv_stage.hpp"

namespace stream {

constexpr int N_STAGES = 9;
constexpr int STAGE_LAYER[N_STAGES] = {0, 2, 4, 6, 8, 10, 12, 13, 14};
constexpr int STAGE_PE[N_STAGES] = {8, 4, 2, 1, 1, 1, 1, 1, 1};
constexpr int STAGE_SIMD[N_STAGES] = {3, 16, 32, 64, 64, 64, 256, 256, 32};

// Registres : par étage, offsets dans les blobs (poids en octets, biais et M0 en indices
// int32), décalage et saturation.
struct StreamDesc {
  int32_t w_off[N_STAGES];
  int32_t b_off[N_STAGES];
  int32_t m0_off[N_STAGES];
  int32_t shift[N_STAGES];
  int32_t qmax[N_STAGES];
};

#ifndef __SYNTHESIS__
extern StageCycles stream_cycles[N_STAGES];
// Si `stream_tap` : copie de la sortie de chaque étage (HWC), pour le testbench.
extern bool stream_tap;
extern std::vector<int8_t> stream_taps[N_STAGES];
#endif

}  // namespace stream

// Entrée (3, 416, 416) et tête (125, 13, 13) en flux HWC.
void yolo_stream(hls::stream<int8_t>& in, hls::stream<int8_t>& out, const int8_t* wts,
                 const int32_t* bias, const int32_t* m0, stream::StreamDesc d);
