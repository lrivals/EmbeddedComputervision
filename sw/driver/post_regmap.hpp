// Registres s_axilite du noyau `yolo_post` (T9.1, hls/kernels/postproc.cpp, bundle
// `control`) : même disposition Vitis HLS que `yolo_conv` (regmap.hpp) — contrôle
// ap_ctrl_hs, 3 pointeurs m_axi 64 bits (un mot réservé après chacun), puis `d` agrégé.
// À vérifier contre `xyolo_post_hw.h` après export (Vitis absent : non vérifié).
#pragma once

#include <cstdint>
#include <cstring>

#include "postproc.hpp"
#include "regmap.hpp"

namespace post_regmap {

using regmap::AP_DONE;
using regmap::AP_IDLE;
using regmap::AP_READY;
using regmap::AP_START;
using regmap::CTRL;
using regmap::GIE;
using regmap::IER;
using regmap::ISR;

constexpr uint32_t ACT = 0x10;
constexpr uint32_t TAB = 0x1c;
constexpr uint32_t RES = 0x28;
constexpr uint32_t D = 0x34;
constexpr int D_WORDS = int(sizeof(accel::PostDesc) / 4);
constexpr uint32_t SPAN = 0x1000;

static_assert(D_WORDS == 6, "PostDesc : 6 × int32");
static_assert(D + 4 * D_WORDS <= SPAN, "registres hors de la fenêtre");

inline void desc_to_words(const accel::PostDesc& d, uint32_t w[D_WORDS]) {
  std::memcpy(w, &d, sizeof d);
}

inline accel::PostDesc words_to_desc(const uint32_t w[D_WORDS]) {
  accel::PostDesc d;
  std::memcpy(&d, w, sizeof d);
  return d;
}

}  // namespace post_regmap
