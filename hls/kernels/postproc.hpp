// Noyau HLS de post-traitement `yolo_post` (T9.1.2, §9.4, §10.3, 2024-zhang).
//
// Un appel traite toutes les têtes d'une image : seuil sur l'entier t_o, tables, coins Q4,
// scores Q16 puis NMS sans tri — mêmes primitives que la référence
// cpp/golden/include/golden/hw_postproc.hpp, donc boîtes égales à `hwpp::run`.
//
// Mémoire (indices, comme `LayerDesc`) :
// - `act` : arène DDR des activations (têtes int8 en place) ;
// - `tab` : mots 32 bits ; à `desc_off`, `nheads` descripteurs de `POST_HEAD_WORDS` mots
//   (champs `PD_*`), les tables σ, e^t, e^{d·s} de la tête h à `PD_LUT_OFF` (3 × 256) ;
// - `res` : à `out_off`, [nombre de boîtes, débordements, puis 6 mots par boîte :
//   x1, y1, x2, y2, score, classe].
#pragma once

#include <cstdint>

#include "golden/hw_postproc.hpp"

namespace accel {

constexpr int POST_MAX_HEADS = 2;
constexpr int POST_CAP = hwpp::CAP;
constexpr int POST_BOX_WORDS = 6;

enum PostHeadField {
  PD_DATA_OFF,     // octets dans l'arène
  PD_GRID_H,       // grille S_h × S_w (non carrée : T11.4)
  PD_GRID_W,
  PD_CLASSES,
  PD_ANCHORS,
  PD_STRIDE_LOG2,
  PD_EXP_FRAC,
  PD_OBJ_THR,
  PD_SOFTMAX,
  PD_LUT_OFF,      // mots dans `tab`
  PD_ANCHOR0,      // 2 × MAX_ANCHORS mots (Q8)
  POST_HEAD_WORDS = PD_ANCHOR0 + 2 * hwpp::MAX_ANCHORS
};

// Registres s_axilite.
struct PostDesc {
  int32_t nheads;
  int32_t desc_off;
  int32_t conf_q;
  int32_t iou_p, iou_q;
  int32_t out_off;
};

#ifndef __SYNTHESIS__
// Estimation de cycles en C-sim (II = 1 sur chaque boucle pipelinée, latences ignorées).
struct PostCycles {
  uint64_t luts = 0;     // copie des tables sur puce
  uint64_t scan = 0;     // lecture de t_o, une cellule par cycle
  uint64_t decode = 0;   // survivantes : 4 + C lectures, max, Σ, scores
  uint64_t nms = 0;      // une comparaison par emplacement occupé, par candidate
  uint64_t store = 0;
  uint64_t total = 0;
  int cells = 0, survivors = 0, candidates = 0;
};
extern PostCycles post_cycles;
#endif

}  // namespace accel

void yolo_post(const int8_t* act, const int32_t* tab, int32_t* res, accel::PostDesc d);
