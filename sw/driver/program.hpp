// Driver du moteur couche par couche (T6.4) : du manifest aux appels du noyau `yolo_conv`.
//
// - Allocation contiguë : les tampons nommés du manifest (A, B, H13, …) sont posés bout à bout
//   dans une seule arène DDR, alignés à 64 octets ; une adresse d'activation est un indice
//   dans cette arène. Une marge (ARENA_SLACK) suit l'arène : le chargeur du noyau lit des mots
//   entiers, jusqu'à 2 mots au-delà de la fin d'une ligne (T10.1).
// - Une **vue** par couche, comme `golden::Engine::place_views` mais en indices : upsample =
//   vue de sa source avec division d'adresse par 2 ; route = concaténation de segments,
//   fusionnés quand ils sont contigus. Aucune copie, aucun noyau dédié (§10.3).
// - Un `LayerDesc` par conv ; maxpool fusionnés dans leur conv. Tuile et pliage choisis par
//   couche (T10.4) ; poids réordonnés une fois dans l'ordre des tuiles (weight_layout.hpp).
//
// Partagé par le testbench HLS (C-sim, co-sim) et l'application ARM (M7).
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include "golden/model.hpp"
#include "accel_config.hpp"
#include "layer_desc.hpp"

namespace driver {

struct Segment {
  int64_t off = 0;  // indice dans l'arène
  int c = 0, h = 0, w = 0;
  int up = 0;
};

struct View {
  Segment seg[accel::MAX_SEGMENTS];
  int nseg = 0;
  int c = 0, h = 0, w = 0;
};

View single_view(int64_t off, int c, int h, int w);
int8_t view_at(const int8_t* arena, const View& v, int ch, int r, int col);
std::vector<int8_t> materialize(const int8_t* arena, const View& v);  // (c, h, w)

// Paramètres du noyau pour la conv `l` lisant `in` ; `prepool_off` < 0 : pas de carte avant
// pooling en DDR. `b_index`, `m0_index` : indices int32 dans le tableau de paramètres ;
// `w_off` : octets dans les poids réordonnés (Program::weights).
accel::LayerDesc conv_desc(const golden::Layer& l, const View& in, int64_t out_off,
                           int64_t prepool_off, int64_t b_index, int64_t m0_index,
                           int64_t w_off);

struct ConvCall {
  int layer = 0;
  accel::LayerDesc desc{};
};

struct Program {
  std::map<std::string, int64_t> base;  // tampon nommé → indice dans l'arène
  int64_t arena_size = 0;
  int64_t input_off = 0;
  std::vector<int32_t> params;          // bias.bin puis requant.bin
  std::vector<int8_t> weights;          // weights.bin dans l'ordre des tuiles (weight_layout)
  std::map<int, int64_t> w_off;         // conv → octet de ses poids dans `weights`
  std::vector<View> views;              // par couche ; nseg = 0 : pas en DDR
  std::vector<ConvCall> calls;          // dans l'ordre d'exécution
};

constexpr int64_t ARENA_ALIGN = 64;
constexpr int64_t ARENA_SLACK = 64;  // ≥ 2 mots du port m_axi

Program build(const golden::Model& m);

// Table du séquenceur (T10.7) : D_WORDS mots int32 par conv, dans l'ordre d'exécution.
std::vector<int32_t> desc_table(const Program& p);

// Signature du top `yolo_conv` (mots de accel::WORD octets).
using Word = std::conditional_t<accel::WORD == 1, uint8_t,
             std::conditional_t<accel::WORD == 2, uint16_t,
             std::conditional_t<accel::WORD == 4, uint32_t, uint64_t>>>;
using Kernel = void (*)(const Word* act_in, Word* act_out, const Word* wts, const int32_t* prm,
                        accel::LayerDesc d, const int32_t* descs, int32_t n_calls);

// Passe avant : copie `input` (C, H, W) dans l'arène puis appelle le noyau pour chaque conv.
// `after` (optionnel) est appelé après chaque conv. L'arène est allouée alignée sur un mot.
void run(const Program& p, const golden::Model& m, std::vector<Word>& arena,
         const int8_t* input, Kernel kernel,
         const std::function<void(const ConvCall&)>& after = nullptr);

}  // namespace driver
