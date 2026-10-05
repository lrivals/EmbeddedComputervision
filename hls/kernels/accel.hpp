// Noyau HLS « moteur unique couche par couche » (§10.1-§10.3, M6).
//
// Un appel de `yolo_conv` exécute une couche conv entière (+ maxpool fusionné) avec le même
// nid de boucles tuilé que le golden (`golden::conv_layer`, cpp/golden/include/golden/conv.hpp) :
// tuiles row → col → to → ti, mêmes bornes de tuiles partielles, même recalcul de la dernière
// ligne en maxpool stride 1. Le résultat est donc égal au golden à l'octet près.
//
// Tuiles : paramètres de compilation (hls/configs/<carte>.tcl → -DACC_TM=…), défauts KV260
// (ADR 0003).
#pragma once

#include <cstdint>

#ifndef ACC_NO_APINT
#include <ap_int.h>
#endif

#include "golden/conv.hpp"
#include "layer_desc.hpp"

#ifndef ACC_TM
#define ACC_TM 32
#endif
#ifndef ACC_TN
#define ACC_TN 24
#endif
#ifndef ACC_TR
#define ACC_TR 13
#endif
#ifndef ACC_TC
#define ACC_TC 13
#endif

namespace accel {

constexpr int TM = ACC_TM, TN = ACC_TN, TR = ACC_TR, TC = ACC_TC;
constexpr int K_MAX = golden::K_MAX;
constexpr int IR = TR + K_MAX - 1;  // S = 1 : S·Tr + K − S
constexpr int IC = TC + K_MAX - 1;
static_assert(TR >= 2 && TC >= 2, "une tuile doit contenir une fenêtre de maxpool 2×2");

#ifndef ACC_NO_APINT
using act_t = ap_int<8>;
using w_t = ap_int<8>;
using acc_t = ap_int<32>;
#else
using act_t = int8_t;  // repli sans en-têtes ap_int : même arithmétique (entiers exacts)
using w_t = int8_t;
using acc_t = int32_t;
#endif

// Tampons sur puce : tailles des formules du §10.2 (vérifiées contre golden/conv.hpp).
static_assert(TN * IR * IC == golden::buf_in_size(TN, TR, TC, K_MAX, 1), "B_in");
static_assert(TM * TN * K_MAX * K_MAX == golden::buf_w_size(TM, TN, K_MAX), "B_w");
static_assert(TM * TR * TC == golden::buf_out_size(TM, TR, TC), "B_out");

// Une tuile de sortie (row, col, to) et ses bornes partielles — mêmes formules que le golden.
struct TileInfo {
  int row, col;        // première ligne / colonne conv (avant pooling)
  int prow0, pcol0;    // première ligne / colonne poolée
  int to;              // premier canal de sortie
  int tr_n, tc_n;      // lignes / colonnes conv calculées
  int np_r, np_c;      // lignes / colonnes poolées produites
  int tm_n;            // canaux de sortie valides
};

#ifndef __SYNTHESIS__
// Estimation de cycles en C-sim (II = 1, profondeurs de pipeline ignorées) ; les cycles réels
// viennent de la co-sim (T6.5).
struct SimCycles {
  uint64_t load_in = 0;     // in_buf : Tn·IR·IC par (tuile, ti)
  uint64_t load_w = 0;      // w_buf : Tm·Tn·K² par (tuile, ti)
  uint64_t compute = 0;     // K²·Tr·Tc par (tuile, ti)
  uint64_t store = 0;       // étage de sortie
  uint64_t sequential = 0;  // somme sans recouvrement
  uint64_t overlapped = 0;  // ping-pong : max(chargement ti+1, calcul ti), max(conv t+1, store t)
};
extern SimCycles sim_cycles;  // remis à zéro à chaque appel de yolo_conv
#endif

}  // namespace accel

// Top (portée globale pour `set_top`) : `act_in` et `act_out` désignent la même arène DDR
// (deux bundles pour que chargement et stockage se recouvrent) ; une conv ne lit jamais ce
// qu'elle écrit (allocation du manifest, docs/conventions.md).
void yolo_conv(const int8_t* act_in, int8_t* act_out, const int8_t* wts, const int32_t* prm,
               accel::LayerDesc d);
