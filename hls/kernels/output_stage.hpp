// Étage de sortie fusionné (T6.3, §9.3, §10.3) : biais + requantification M0/décalage +
// leaky 13/128 + saturation, puis maxpool 2×2 en stride 2 ou 1 (réplication du bord bas et
// droit) ou 5×5 en stride 1 (SPP de YOLOv7-tiny, T26.0 : fenêtre centrée, réplication des
// quatre bords), et carte avant pooling en DDR si demandée (prépool de L08 pour la route 20).
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
inline int imax(int a, int b) { return a > b ? a : b; }

// Carte avant pooling écrite par la tuile `t` : ses lignes et colonnes propres (hors
// recouvrement), à partir de own_row0 / own_col0 dans la tuile ; en stride 1, celles de ses
// lignes poolées. Mêmes formules que le golden (`golden::conv_layer`) et tools/perf_model.py.
inline int own_row0(const LayerDesc& d, const TileInfo& t) { return t.prow0 * d.ps() - t.row; }
inline int own_col0(const LayerDesc& d, const TileInfo& t) { return t.pcol0 * d.ps() - t.col; }
inline int own_rows(const LayerDesc& d, const TileInfo& t) {
  return d.ps() == 1 ? t.np_r : imin(t.tr_n, ((d.tr - d.pk()) / d.ps() + 1) * d.ps());
}
inline int own_cols(const LayerDesc& d, const TileInfo& t) {
  return d.ps() == 1 ? t.np_c : imin(t.tc_n, ((d.tc - d.pk()) / d.ps() + 1) * d.ps());
}

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

// ABFT (T25.3) : vérification de la tuile (row, col) à laquelle appartient la tuile stockée.
// Σ des tm_n voies + biais, accumulée sur les tuiles `to` de (row, col) dans `sum_buf`
// (modulo 2³², comme golden::conv_layer), comparée à la voie de contrôle `cs_buf` + Σ_f b_f
// au dernier `to`, pour les tr_n × tc_n pixels calculés. Dans la passe de requantification :
// pas de cycle de plus (additionneur de RQ voies et comparateur en plus).
struct AbftTile {
  bool on = false;     // couche vérifiée (somme de contrôle exportée)
  bool first = false;  // premier `to` de (row, col) : sum_buf repart de zéro
  bool last = false;   // dernier `to` : comparaison
  int32_t cs_b = 0;    // Σ_f b_f
};

// Stocke la tuile `t` de `out_buf` (accumulateurs int32 sans biais) en DDR.
inline void store_tile(word_t* act_out, const int32_t* prm, const LayerDesc& d,
                       const TileInfo& t, const acc_t out_buf[TM][TRB][TCB],
                       const AbftTile& ab, const uint32_t cs_buf[TRB][TCB],
                       uint32_t sum_buf[TRB][TCB], int32_t& errors) {
#pragma HLS INLINE off
  const int R = d.out_h(), C = d.out_w();
  const int pk = d.pk(), ps = d.ps(), lo = d.pool_lo();
  const int Rp = d.pool_h(), Cp = d.pool_w();
  const int tr = d.tr, tc = d.tc;
  // Lignes et colonnes propres à la tuile (hors recouvrement du maxpool stride 1).
  const int r0 = own_row0(d, t), c0 = own_col0(d, t);
  const int own_r = own_rows(d, t), own_c = own_cols(d, t);
  const bool write_prepool = d.pooled() && d.prepool_off >= 0;

  int8_t q[RQ][TRB][TCB];  // tuiles requantifiées d'un groupe de RQ canaux
#pragma HLS ARRAY_PARTITION variable=q complete dim=1
#pragma HLS ARRAY_PARTITION variable=q cyclic factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=q cyclic factor=2 dim=3
  int8_t row[IC];
#pragma HLS ARRAY_PARTITION variable=row complete
  // SiLU (T26.2) : table int8 de 256 entrées, 64 mots juste après les M0 de la couche (lue ici
  // à chaque tuile en C-sim ; une fois par couche en matériel, T26.3).
  int8_t silu_t[256];
#pragma HLS ARRAY_PARTITION variable=silu_t complete
  if (d.leaky == 2)
    for (int k = 0; k < 256; ++k)
      silu_t[k] = int8_t(uint32_t(prm[d.m0_off + d.cout + k / 4]) >> (8 * (k % 4)));

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
      uint32_t sum = 0;
      for (int r = 0; r < RQ; ++r) {
#pragma HLS UNROLL
        int32_t y = golden::requantize(int32_t(out_buf[g + r][trr][tcc]) + bias[r], m0[r],
                                       d.shift);
        if (d.leaky == 2) {
          q[r][trr][tcc] = golden::silu_lookup(y, silu_t);
        } else {
          if (d.leaky) y = golden::leaky_int(y);
          q[r][trr][tcc] = golden::clip_q(y, d.qmax);
        }
        if (g + r < t.tm_n) sum += uint32_t(int32_t(out_buf[g + r][trr][tcc])) + uint32_t(bias[r]);
      }
      if (ABFT && ab.on) {
        const uint32_t tot = (ab.first && g == 0 ? 0u : sum_buf[trr][tcc]) + sum;
        sum_buf[trr][tcc] = tot;
        if (ab.last && g + RQ >= t.tm_n && trr < t.tr_n && tcc < t.tc_n &&
            tot != cs_buf[trr][tcc] + uint32_t(ab.cs_b))
          ++errors;
      }
    }
#ifndef __SYNTHESIS__
    // T25.7 : bit inversé dans la tuile requantifiée que lit le maxpool fusionné.
    if (hls_fault.armed(FB_Q, t.id) && hls_fault.a >= g && hls_fault.a < g + RQ) {
      flip8(q[hls_fault.a - g][hls_fault.b][hls_fault.c], hls_fault.bit);
      hls_fault.fired = 1;
    }
#endif

  store_ch:
    for (int r = 0; r < RQ; ++r) {
      if (g + r >= t.tm_n) break;
      const int o = t.to + g + r;
      if (write_prepool) {
      prepool:
        for (int trr = r0; trr < r0 + own_r; ++trr) {
          for (int tcc = 0; tcc < TCB; ++tcc)
            row[tcc] = tcc < own_c && c0 + tcc < TCB ? q[r][trr][c0 + tcc] : 0;
          write_row(act_out, d.prepool_off + (int64_t(o) * R + t.row + trr) * C + t.col + c0,
                    row, own_c);
        }
      }

    pool:
      for (int pr = 0; pr < t.np_r; ++pr) {
        for (int pc = 0; pc < IC; ++pc) {
#pragma HLS UNROLL
          int8_t m = -128;
          for (int a = 0; a < PK_MAX; ++a)
            for (int c = 0; c < PK_MAX; ++c) {
              if (pc < t.np_c && a < pk && c < pk) {
                // Fenêtre bornée à la carte : réplication du bord (stride 1, §10.3) ;
                // centrée pour le 5×5 (lo = 2, T26.0).
                const int rr = imin(imax((t.prow0 + pr) * ps - lo + a, 0), R - 1) - t.row;
                const int s = imin(imax((t.pcol0 + pc) * ps - lo + c, 0), C - 1) - t.col;
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

// Sans vérification ABFT (testbench de l'étage de sortie, T28.1).
inline void store_tile(word_t* act_out, const int32_t* prm, const LayerDesc& d,
                       const TileInfo& t, const acc_t out_buf[TM][TRB][TCB]) {
  static uint32_t none[TRB][TCB];
  int32_t errors = 0;
  store_tile(act_out, prm, d, t, out_buf, AbftTile{}, none, none, errors);
}

}  // namespace accel
