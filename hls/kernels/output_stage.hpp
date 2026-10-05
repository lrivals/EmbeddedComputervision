// Étage de sortie fusionné (T6.3, §9.3, §10.3) : biais + requantification M0/décalage +
// leaky 13/128 + saturation, puis maxpool 2×2 en stride 2 ou 1 (réplication du bord bas et
// droit), et carte avant pooling en DDR si demandée (prépool de L08 pour la route 20).
//
// L'arithmétique est celle du golden (`golden::requantize`, `leaky_int`, `clip_q`), réutilisée
// telle quelle : produit acc·M0 sur 64 bits → 4 DSP48 (32 × 31 bits) par multiplieur. RQ
// multiplieurs en parallèle (T10.2) : RQ canaux requantifiés par cycle, Tr·Tc cycles par
// groupe de RQ canaux.
//
// Écritures en mots de WORD octets (T10.1), ligne par ligne : une ligne de n octets coûte
// row_words(n) mots, quel que soit son alignement. Les mots de bord ne sont que partiellement
// écrits : en matériel, masque d'octets (WSTRB) du port m_axi — à confirmer en synthèse ; en
// C-sim, fusion avec le contenu de la DDR.
#pragma once

#include "accel.hpp"

namespace accel {

inline int imin(int a, int b) { return a < b ? a : b; }

// Écrit les n octets de `src` à l'adresse d'octet `addr` : row_words(n) mots.
inline void write_row(word_t* mem, int64_t addr, const int8_t src[IC], int n) {
  const int64_t w0 = addr / WORD;
  const int nw = row_words(n);
write_words:
  for (int k = 0; k < nw; ++k) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT max=3
    word_t v = 0, keep = 0;  // keep : octets du mot hors de la ligne
    int used = 0;
    for (int b = 0; b < WORD; ++b) {
#pragma HLS UNROLL
      const int64_t q = (w0 + k) * WORD + b - addr;
      if (q >= 0 && q < n) {
        v |= word_t(uint8_t(src[q])) << (8 * b);
        ++used;
      } else {
        keep |= word_t(0xff) << (8 * b);
      }
    }
    if (used == WORD) mem[w0 + k] = v;
    else if (used > 0) mem[w0 + k] = (mem[w0 + k] & keep) | v;  // WSTRB partiel
  }
}

// Stocke la tuile `t` de `out_buf` (accumulateurs int32 sans biais) en DDR.
inline void store_tile(word_t* act_out, const int32_t* prm, const LayerDesc& d,
                       const TileInfo& t, const acc_t out_buf[TM][TRB][TCB]) {
#pragma HLS INLINE off
  const int R = d.out_h(), C = d.out_w();
  const int pk = d.pk(), ps = d.ps();
  const int Rp = d.pool_h(), Cp = d.pool_w();
  const int tr = d.tr, tc = d.tc;
  const int Pr = (tr - pk) / ps + 1, Pc = (tc - pk) / ps + 1;
  // Lignes et colonnes propres à la tuile (hors recouvrement du maxpool stride 1).
  const int own_r = imin(t.tr_n, Pr * ps), own_c = imin(t.tc_n, Pc * ps);
  const bool write_prepool = d.pooled() && d.prepool_off >= 0;

  int8_t q[RQ][TRB][TCB];  // tuiles requantifiées d'un groupe de RQ canaux
#pragma HLS ARRAY_PARTITION variable=q complete dim=1
#pragma HLS ARRAY_PARTITION variable=q cyclic factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=q cyclic factor=2 dim=3
  int8_t row[IC];
#pragma HLS ARRAY_PARTITION variable=row complete

store_group:
  for (int g = 0; g < TM; g += RQ) {
    if (g >= t.tm_n) break;
    int32_t bias[RQ], m0[RQ];
    for (int r = 0; r < RQ; ++r) {
      const int o = imin(t.to + g + r, d.cout - 1);
      bias[r] = prm[d.b_off + o];
      m0[r] = prm[d.m0_off + o];
    }

  requant:
    for (int n = 0; n < tr * tc; ++n) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT max=TRB*TCB
      const int trr = n / tc, tcc = n % tc;
      for (int r = 0; r < RQ; ++r) {
#pragma HLS UNROLL
        int32_t y = golden::requantize(int32_t(out_buf[g + r][trr][tcc]) + bias[r], m0[r],
                                       d.shift);
        if (d.leaky) y = golden::leaky_int(y);
        q[r][trr][tcc] = golden::clip_q(y, d.qmax);
      }
    }

  store_ch:
    for (int r = 0; r < RQ; ++r) {
      if (g + r >= t.tm_n) break;
      const int o = t.to + g + r;
      if (write_prepool) {
      prepool:
        for (int trr = 0; trr < own_r; ++trr) {
          for (int tcc = 0; tcc < TCB; ++tcc) row[tcc] = tcc < own_c ? q[r][trr][tcc] : 0;
          write_row(act_out, d.prepool_off + (int64_t(o) * R + t.row + trr) * C + t.col, row,
                    own_c);
        }
      }

    pool:
      for (int pr = 0; pr < t.np_r; ++pr) {
        for (int pc = 0; pc < IC; ++pc) {
#pragma HLS UNROLL
          int8_t m = -128;
          for (int a = 0; a < 2; ++a)
            for (int c = 0; c < 2; ++c) {
              if (pc < t.np_c && a < pk && c < pk) {
                // Fenêtre bornée à la carte : réplication du bord (stride 1, §10.3).
                const int rr = imin((t.prow0 + pr) * ps + a, R - 1) - t.row;
                const int s = imin((t.pcol0 + pc) * ps + c, C - 1) - t.col;
                const int8_t v = q[r][rr][s];
                m = v > m ? v : m;
              }
            }
          row[pc] = m;
        }
        write_row(act_out, d.out_off + (int64_t(o) * Rp + t.prow0 + pr) * Cp + t.pcol0, row,
                  t.np_c);
      }
    }
  }
}

}  // namespace accel
