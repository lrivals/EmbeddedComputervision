// Noyau HLS « moteur unique couche par couche » (§10.1-§10.3, M6, M10).
//
// Un appel de `yolo_conv` exécute une couche conv entière (+ maxpool fusionné) avec un nid de
// boucles tuilé comme le golden (`golden::conv_layer`, cpp/golden/include/golden/conv.hpp) :
// tuiles row → col → to → ti, tuiles partielles, recalcul de la dernière ligne en maxpool
// stride 1. La tuile (d.tr, d.tc) et le pliage (d.fold) sont choisis par couche par le driver
// (T10.4) ; une somme d'entiers ne dépendant pas de l'ordre, le résultat est égal au golden à
// l'octet près quelle que soit la tuile.
//
// Configuration (tuiles, largeur des ports, trim, requant, pliage) : accel_config.hpp.
// Séquenceur (T10.7) : avec n_calls > 0, le noyau lit n_calls descripteurs en DDR et enchaîne
// les couches sans l'ARM (une seule fin, une seule IRQ).
#pragma once

#include <cstdint>

#ifndef ACC_NO_APINT
#include <ap_int.h>
#endif

#include "accel_config.hpp"
#include "golden/conv.hpp"
#include "layer_desc.hpp"
#include "weight_layout.hpp"

namespace accel {

static_assert(K_MAX == golden::K_MAX, "K_MAX");
static_assert(TR >= 2 && TC >= 2, "une tuile doit contenir une fenêtre de maxpool 2×2");

// Mot des ports m_axi act_in, act_out, wts : WORD octets, petit-boutiste (octet b = bits
// 8b … 8b + 7), comme la DDR vue par l'ARM et le PC.
template <int B> struct WordOf;
template <> struct WordOf<1> { using type = uint8_t; };
template <> struct WordOf<2> { using type = uint16_t; };
template <> struct WordOf<4> { using type = uint32_t; };
template <> struct WordOf<8> { using type = uint64_t; };
using word_t = WordOf<WORD>::type;

inline int8_t word_byte(word_t w, int b) { return int8_t(uint8_t(w >> (8 * b))); }

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
static_assert(TN * IR * IC == golden::buf_in_size(TN, TRB, TCB, K_MAX, 1), "B_in");
static_assert(TM * TN * K_MAX * K_MAX == golden::buf_w_size(TM, TN, K_MAX), "B_w");
static_assert(TM * TRB * TCB == golden::buf_out_size(TM, TRB, TCB), "B_out");

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
  uint64_t load_in = 0;     // in_buf : voies × lignes × mots par ligne, par (tuile, ti)
  uint64_t load_w = 0;      // w_buf : mots du bloc de poids, par (tuile, ti)
  uint64_t compute = 0;     // kh·k·Tr·Tc par (tuile, ti)
  uint64_t store = 0;       // étage de sortie
  uint64_t sequential = 0;  // somme sans recouvrement
  uint64_t overlapped = 0;  // ping-pong : max(chargement ti+1, calcul ti), max(conv t+1, store t)
};
extern SimCycles sim_cycles;  // remis à zéro à chaque appel de yolo_conv (cumul du séquenceur)
#endif

// Produit poids × activation de la PE. Mode `ACC_WMODE_POW2` (T9.2.4, REQ-YOLO) : poids sur
// les niveaux « mixed powers-of-two » ±(2^a + 2^{a−k}) (python/yolo/quant/pow2.py), le
// produit est fait par deux décalages et une addition, sans DSP. Le poids arrive en int8 et
// est décodé en (signe, a, k) au chargement dans w_buf (en matériel : stockage des codes 6
// bits) ; résultat égal au produit tant que le poids est sur un niveau (le testbench le
// vérifie sur le modèle tools/quant_lowbit.py --weights pow2).
#ifdef ACC_WMODE_POW2
inline int32_t mul_w(int w, int x) {
  const int m = w < 0 ? -w : w;
  if (m == 0) return 0;
  int a = 0;
  while ((2 << a) <= m) ++a;  // bit de poids fort
  const int rest = m - (1 << a);
  int b = 0;
  while (rest && (2 << b) <= rest) ++b;
  const int32_t p = (int32_t(x) << a) + (rest ? (int32_t(x) << b) : 0);
  return w < 0 ? -p : p;
}
#else
inline int32_t mul_w(int w, int x) { return w * x; }
#endif

}  // namespace accel

// Top (portée globale pour `set_top`) : `act_in` et `act_out` désignent la même arène DDR
// (deux bundles pour que chargement et stockage se recouvrent) ; une conv ne lit jamais ce
// qu'elle écrit (allocation du manifest, docs/conventions.md). Arène et poids alignés sur un
// mot ; le chargeur lit jusqu'à 2 mots au-delà d'une ligne, d'où une marge après l'arène
// (driver::ARENA_SLACK). `wts` : poids dans l'ordre des tuiles (weight_layout.hpp).
// n_calls = 0 : une couche, décrite par `d` ; n_calls > 0 : `descs` contient n_calls
// descripteurs de D_WORDS mots, exécutés dans l'ordre (`d` ignoré).
void yolo_conv(const accel::word_t* act_in, accel::word_t* act_out, const accel::word_t* wts,
               const int32_t* prm, accel::LayerDesc d, const int32_t* descs, int32_t n_calls);
