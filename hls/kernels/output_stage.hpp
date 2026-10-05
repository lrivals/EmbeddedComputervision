// Étage de sortie fusionné (T6.3, §9.3, §10.3) : biais + requantification M0/décalage +
// leaky 13/128 + saturation, puis maxpool 2×2 en stride 2 ou 1 (réplication du bord bas et
// droit), et carte avant pooling en DDR si demandée (prépool de L08 pour la route 20).
//
// L'arithmétique est celle du golden (`golden::requantize`, `leaky_int`, `clip8`), réutilisée
// telle quelle : produit acc·M0 sur 64 bits → 4 DSP48 (32 × 31 bits) pour l'unique
// multiplieur de requantification, partagé par les Tm canaux (une valeur par cycle).
#pragma once

#include "accel.hpp"

namespace accel {

inline int imin(int a, int b) { return a < b ? a : b; }

// Stocke la tuile `t` de `out_buf` (accumulateurs int32 sans biais) en DDR.
inline void store_tile(int8_t* act_out, const int32_t* prm, const LayerDesc& d,
                       const TileInfo& t, const acc_t out_buf[TM][TR][TC]) {
#pragma HLS INLINE off
  const int R = d.out_h(), C = d.out_w();
  const int pk = d.pk(), ps = d.ps();
  const int Rp = d.pool_h(), Cp = d.pool_w();
  const int Pr = (TR - pk) / ps + 1, Pc = (TC - pk) / ps + 1;
  // Lignes et colonnes propres à la tuile (hors recouvrement du maxpool stride 1).
  const int own_r = imin(t.tr_n, Pr * ps), own_c = imin(t.tc_n, Pc * ps);
  const bool write_prepool = d.pooled() && d.prepool_off >= 0;

  int8_t q[TR][TC];  // tuile requantifiée d'un canal
#pragma HLS ARRAY_PARTITION variable=q cyclic factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable=q cyclic factor=2 dim=2

store_ch:
  for (int too = 0; too < TM; ++too) {
    if (too >= t.tm_n) break;
    const int o = t.to + too;
    const int32_t bias = prm[d.b_off + o];
    const int32_t m0 = prm[d.m0_off + o];

  requant:
    for (int n = 0; n < TR * TC; ++n) {
#pragma HLS PIPELINE II=1
      const int trr = n / TC, tcc = n % TC;
      int32_t y = golden::requantize(int32_t(out_buf[too][trr][tcc]) + bias, m0, d.shift);
      if (d.leaky) y = golden::leaky_int(y);
      q[trr][tcc] = golden::clip8(y);
    }

    if (write_prepool) {
    prepool:
      for (int trr = 0; trr < own_r; ++trr)
        for (int tcc = 0; tcc < own_c; ++tcc) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT max=TC
          act_out[d.prepool_off + (o * R + t.row + trr) * C + t.col + tcc] = q[trr][tcc];
        }
    }

  pool:
    for (int pr = 0; pr < t.np_r; ++pr)
      for (int pc = 0; pc < t.np_c; ++pc) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT max=TC
        int8_t m = -128;
        for (int a = 0; a < 2; ++a)
          for (int c = 0; c < 2; ++c) {
#pragma HLS UNROLL
            if (a < pk && c < pk) {
              // Fenêtre bornée à la carte : réplication du bord (stride 1, §10.3).
              const int r = imin((t.prow0 + pr) * ps + a, R - 1) - t.row;
              const int s = imin((t.pcol0 + pc) * ps + c, C - 1) - t.col;
              const int8_t v = q[r][s];
              m = v > m ? v : m;
            }
          }
        act_out[d.out_off + (o * Rp + t.prow0 + pr) * Cp + t.pcol0 + pc] = m;
      }
  }
}

}  // namespace accel
