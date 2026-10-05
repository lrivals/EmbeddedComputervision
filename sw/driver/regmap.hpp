// Registres s_axilite du noyau `yolo_conv` (bundle `control`, T7.2).
//
// Disposition produite par Vitis HLS pour le top de conv_pe.cpp : bloc de contrôle
// ap_ctrl_hs, puis les 4 pointeurs m_axi (64 bits, `config_interface -m_axi_addr64`, un mot
// réservé après chacun), puis `d` agrégé (`#pragma HLS AGGREGATE`) : 26 champs int32, le
// champ i au mot i (premier champ = bits de poids faible). À vérifier contre l'en-tête généré
// `xyolo_conv_hw.h` après `make hls-export` : `make check-regmap` (tools/check_regmap.py).
#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "layer_desc.hpp"

namespace regmap {

constexpr uint32_t CTRL = 0x00;  // bit 0 ap_start, 1 ap_done (COR), 2 ap_idle, 3 ap_ready
constexpr uint32_t GIE = 0x04;   // activation globale de l'interruption
constexpr uint32_t IER = 0x08;   // bit 0 : ap_done
constexpr uint32_t ISR = 0x0c;   // bit 0 : ap_done (basculé par écriture de 1)
constexpr uint32_t ACT_IN = 0x10;
constexpr uint32_t ACT_OUT = 0x1c;
constexpr uint32_t WTS = 0x28;
constexpr uint32_t PRM = 0x34;
constexpr uint32_t D = 0x40;

constexpr uint32_t AP_START = 1u << 0;
constexpr uint32_t AP_DONE = 1u << 1;
constexpr uint32_t AP_IDLE = 1u << 2;
constexpr uint32_t AP_READY = 1u << 3;

constexpr int D_WORDS = int(sizeof(accel::LayerDesc) / 4);
constexpr uint32_t END = D + 4 * D_WORDS;  // premier octet après `d`
constexpr uint32_t SPAN = 0x1000;          // fenêtre AXI-Lite (assign_bd_address, 4 Ko)

static_assert(D_WORDS == 26 && sizeof(accel::LayerDesc) == 4 * 26, "LayerDesc : 26 × int32");
static_assert(std::is_trivially_copyable<accel::LayerDesc>::value, "LayerDesc copiable");
static_assert(END <= SPAN, "registres hors de la fenêtre");

inline void desc_to_words(const accel::LayerDesc& d, uint32_t w[D_WORDS]) {
  std::memcpy(w, &d, sizeof d);
}

inline accel::LayerDesc words_to_desc(const uint32_t w[D_WORDS]) {
  accel::LayerDesc d;
  std::memcpy(&d, w, sizeof d);
  return d;
}

}  // namespace regmap
