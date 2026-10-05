// Registres du streaming (T10.9) : noyau `yolo_stream` (bundle `control`) et AXI DMA en mode
// direct (PG021) qui lui apporte l'entrée (MM2S) et rapporte la tête (S2MM).
//
// `yolo_stream` : même disposition Vitis HLS que `yolo_conv` (regmap.hpp) — contrôle
// ap_ctrl_hs, 3 pointeurs m_axi 64 bits (un mot réservé après chacun), puis `d` agrégé
// (StreamDesc, 45 mots). Les ports AXI-Stream n'ont pas de registre. À vérifier contre
// `xyolo_stream_hw.h` après `make hls-export-stream` (Vitis absent : non vérifié).
#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "regmap.hpp"
#include "stream_desc.hpp"

namespace stream_regmap {

using regmap::AP_DONE;
using regmap::AP_IDLE;
using regmap::AP_READY;
using regmap::AP_START;
using regmap::CTRL;
using regmap::GIE;
using regmap::IER;
using regmap::ISR;

constexpr uint32_t WTS = 0x10;
constexpr uint32_t BIAS = 0x1c;
constexpr uint32_t M0 = 0x28;
constexpr uint32_t D = 0x34;
constexpr int D_WORDS = stream::S_WORDS;
constexpr uint32_t SPAN = 0x1000;
static_assert(D + 4 * D_WORDS <= SPAN, "registres hors de la fenêtre");
static_assert(std::is_trivially_copyable<stream::StreamDesc>::value, "StreamDesc copiable");

inline void desc_to_words(const stream::StreamDesc& d, uint32_t w[D_WORDS]) {
  std::memcpy(w, &d, sizeof d);
}

inline stream::StreamDesc words_to_desc(const uint32_t w[D_WORDS]) {
  stream::StreamDesc d;
  std::memcpy(&d, w, sizeof d);
  return d;
}

}  // namespace stream_regmap

// AXI DMA (LogiCORE PG021), mode direct, adresses 64 bits.
namespace dma_regmap {

constexpr uint32_t MM2S_DMACR = 0x00;
constexpr uint32_t MM2S_DMASR = 0x04;
constexpr uint32_t MM2S_SA = 0x18;
constexpr uint32_t MM2S_SA_MSB = 0x1c;
constexpr uint32_t MM2S_LENGTH = 0x28;  // écrire la longueur lance le transfert
constexpr uint32_t S2MM_DMACR = 0x30;
constexpr uint32_t S2MM_DMASR = 0x34;
constexpr uint32_t S2MM_DA = 0x48;
constexpr uint32_t S2MM_DA_MSB = 0x4c;
constexpr uint32_t S2MM_LENGTH = 0x58;  // octets reçus (jusqu'à TLAST) après la fin
constexpr uint32_t SPAN = 0x1000;

constexpr uint32_t CR_RS = 1u << 0;          // marche
constexpr uint32_t CR_RESET = 1u << 2;
constexpr uint32_t CR_IOC_IRQ_EN = 1u << 12;
constexpr uint32_t CR_ERR_IRQ_EN = 1u << 14;
constexpr uint32_t SR_HALTED = 1u << 0;
constexpr uint32_t SR_IDLE = 1u << 1;
constexpr uint32_t SR_ERR = 0x70u;           // DMAIntErr | DMASlvErr | DMADecErr
constexpr uint32_t SR_IOC_IRQ = 1u << 12;    // fin de transfert (effacé par écriture de 1)
constexpr uint32_t SR_ERR_IRQ = 1u << 14;

}  // namespace dma_regmap
