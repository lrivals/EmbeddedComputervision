// Tiny-YOLOv2 en architecture streaming (T9.4.2, T10.8) : 9 étages conv chaînés par des flux
// sous DATAFLOW. Repliements PE × SIMD : plan de tools/stream_model.py pour la KV260 (poids 8
// bits) ; L12 et L13 gardent leur carte d'entrée et lisent leurs poids en DDR (hybride), les
// autres étages lisent les leurs en ROM sur la puce (STREAM_ROM, tools/gen_stream_rom.py).
#pragma once

#include <cstdint>
#include <vector>

#include "conv_stage.hpp"
#include "stream_desc.hpp"

namespace stream {

constexpr int STAGE_PE[N_STAGES] = {8, 4, 2, 1, 1, 1, 1, 1, 1};
constexpr int STAGE_SIMD[N_STAGES] = {3, 16, 32, 64, 64, 64, 256, 256, 32};
// Poids en DDR, carte d'entrée entière et ordre PE extérieur (T10.8).
constexpr bool STAGE_FRAME[N_STAGES] = {false, false, false, false, false,
                                        false, true,  true,  false};
// Sortie canal par canal vers l'étage FRAME suivant (L12 → L13), sinon HWC.
constexpr bool STAGE_OUT_CHW[N_STAGES] = {false, false, false, false, false,
                                          false, true,  false, false};
// Profondeur du flux qui suit l'étage k (une ligne de sortie, stream_model.fifo_depths) ;
// reprise littéralement dans les `#pragma HLS STREAM` de yolo_stream.cpp.
constexpr int FIFO_DEPTH[N_STAGES - 1] = {3328, 3328, 3328, 3328, 3328, 6656, 13312, 13312};

#ifndef __SYNTHESIS__
extern StageCycles stream_cycles[N_STAGES];
// Si `stream_tap` : copie de la sortie de chaque étage (HWC, ou CHW si STAGE_OUT_CHW).
extern bool stream_tap;
extern std::vector<int8_t> stream_taps[N_STAGES];
#endif

}  // namespace stream

// Ports AXI-Stream d'un octet avec TLAST (DMA S2MM, T10.9) : entrée HWC, TLAST sur la
// dernière valeur de la tête.
#if defined(__SYNTHESIS__) || defined(__VITIS_HLS__)
#include <ap_axi_sdata.h>
typedef hls::axis<ap_int<8>, 0, 0, 0> axis_byte;
#else
struct axis_byte {
  int8_t data = 0;
  bool last = false;
};
#endif

void yolo_stream(hls::stream<axis_byte>& in, hls::stream<axis_byte>& out, const int8_t* wts,
                 const int32_t* bias, const int32_t* m0, stream::StreamDesc d);
