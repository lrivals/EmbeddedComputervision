// Étage streaming d'une conv (T9.4.2, §10.1) : un bloc matériel par couche
// [2018-venieris#004.0], activations en flux HWC (pixel par pixel, canaux contigus).
//
// - Fenêtre glissante par line buffer [2023-montgomerie-corcoran#011.0] : K lignes
//   circulaires de W·C_in mots (les K − 1 lignes du §10.2 et la ligne en cours d'arrivée) ;
//   la ligne de sortie r est calculée dès que la ligne d'entrée r + pad est arrivée.
// - Mode `FRAME` (poids en DDR, hybride de tools/stream_model.py) : toute la carte d'entrée
//   est gardée sur la puce. En matériel, la boucle des groupes PE passe à l'extérieur pour
//   ne lire chaque poids qu'une fois par image ; la C-sim garde l'ordre du line buffer (le
//   résultat et le nombre d'itérations ne dépendent pas de l'ordre).
// - Repliement PE × SIMD : PE canaux de sortie et SIMD canaux d'entrée par cycle ; une
//   « itération » = (pixel, groupe PE, groupe SIMD, position du noyau). Les cycles C-sim
//   comptent ces itérations (II = 1) : égaux à `stage_cycles` du modèle.
// - Étage de sortie : arithmétique du golden (`golden::requantize`, `leaky_int`, `clip_q`),
//   maxpool 2×2 fusionné en stride 2, ou en stride 1 avec réplication de la dernière ligne et
//   de la dernière colonne (§10.3, non traité dans la base).
#pragma once

#include <cstdint>

#include "golden/conv.hpp"
#include "stream_shim.hpp"

namespace stream {

struct StageParams {
  const int8_t* w = nullptr;      // (C_out, C_in, K, K)
  const int32_t* bias = nullptr;  // C_out
  const int32_t* m0 = nullptr;    // C_out
  int shift = 31;
  int qmax = 127;
  bool leaky = true;
};

#ifndef __SYNTHESIS__
struct StageCycles {
  uint64_t mac = 0;  // itérations PE × SIMD
};
#endif

// Maxpool fusionné en flux : reçoit les lignes de sortie conv (W × C, HWC) une à une.
template <int H, int W, int C, int PK, int PS>
struct PoolUnit {
  int8_t prev[W][C];
  int rows = 0;

  static constexpr int OH = PK == 0 ? H : (PS == 1 ? H : (H - PK) / PS + 1);
  static constexpr int OW = PK == 0 ? W : (PS == 1 ? W : (W - PK) / PS + 1);

  void emit_pooled(const int8_t a[W][C], const int8_t b[W][C], hls::stream<int8_t>& out) {
    for (int pc = 0; pc < OW; ++pc) {
      const int c0 = pc * PS, c1 = c0 + 1 < W ? c0 + 1 : W - 1;  // réplication à droite
      for (int ch = 0; ch < C; ++ch) {
        int8_t m = a[c0][ch];
        m = a[c1][ch] > m ? a[c1][ch] : m;
        m = b[c0][ch] > m ? b[c0][ch] : m;
        m = b[c1][ch] > m ? b[c1][ch] : m;
        out.write(m);
      }
    }
  }

  void push(const int8_t row[W][C], hls::stream<int8_t>& out) {
    if (PK == 0) {
      for (int c = 0; c < W; ++c)
        for (int ch = 0; ch < C; ++ch) out.write(row[c][ch]);
      return;
    }
    const int r = rows++;
    if (PS == 2) {
      if (r % 2 == 1 && r / 2 < OH) emit_pooled(prev, row, out);
    } else if (r > 0) {
      emit_pooled(prev, row, out);  // ligne poolée r − 1 = max(conv r − 1, conv r)
    }
    for (int c = 0; c < W; ++c)
      for (int ch = 0; ch < C; ++ch) prev[c][ch] = row[c][ch];
    if (PS == 1 && rows == H) emit_pooled(prev, prev, out);  // dernière ligne répliquée
  }
};

// Une sortie conv (pixel, C_out canaux) à partir d'une fenêtre lue par `px(ci, i, j)`.
template <int K, int CIN, int COUT, int PE, int SIMD, class Px>
inline void conv_pixel(const StageParams& p, Px px, int8_t out[COUT]
#ifndef __SYNTHESIS__
                       , StageCycles& cyc
#endif
) {
  static_assert(COUT % PE == 0 && CIN % SIMD == 0, "repliement : PE | C_out, SIMD | C_in");
  for (int og = 0; og < COUT; og += PE) {
    int32_t acc[PE] = {};
    for (int ig = 0; ig < CIN; ig += SIMD)
      for (int i = 0; i < K; ++i)
        for (int j = 0; j < K; ++j) {
#pragma HLS PIPELINE II=1
#ifndef __SYNTHESIS__
          ++cyc.mac;
#endif
          for (int pe = 0; pe < PE; ++pe)
            for (int s = 0; s < SIMD; ++s) {
              const int o = og + pe, c = ig + s;
              acc[pe] += int32_t(p.w[((o * CIN + c) * K + i) * K + j]) * px(c, i, j);
            }
        }
    for (int pe = 0; pe < PE; ++pe) {
      const int o = og + pe;
      int32_t y = golden::requantize(acc[pe] + p.bias[o], p.m0[o], p.shift);
      if (p.leaky) y = golden::leaky_int(y);
      out[o] = golden::clip_q(y, p.qmax);
    }
  }
}

template <int K, int CIN, int COUT, int H, int W, int PE, int SIMD, int PK, int PS,
          bool FRAME>
void conv_stage(hls::stream<int8_t>& in, hls::stream<int8_t>& out, const StageParams& p
#ifndef __SYNTHESIS__
                , StageCycles& cyc
#endif
) {
  constexpr int PAD = K / 2;
  constexpr int ROWS = FRAME ? H : K;  // carte entière ou line buffer circulaire
  static int8_t buf[ROWS][W][CIN];
  static int8_t row[W][COUT];
  static PoolUnit<H, W, COUT, PK, PS> pool;
  pool.rows = 0;
  int read = 0;
  auto read_row = [&]() {
    for (int c = 0; c < W; ++c)
      for (int ch = 0; ch < CIN; ++ch) buf[read % ROWS][c][ch] = in.read();
    ++read;
  };
  if (FRAME)
    while (read < H) read_row();
  for (int r = 0; r < H; ++r) {
    while (read <= (r + PAD < H ? r + PAD : H - 1)) read_row();
    for (int c = 0; c < W; ++c) {
      auto px = [&](int ch, int i, int j) -> int32_t {
        const int ir = r - PAD + i, ic = c - PAD + j;
        if (ir < 0 || ir >= H || ic < 0 || ic >= W) return 0;  // complétion par des zéros
        return buf[ir % ROWS][ic][ch];
      };
      int8_t o[COUT];
      conv_pixel<K, CIN, COUT, PE, SIMD>(p, px, o
#ifndef __SYNTHESIS__
                                         , cyc
#endif
      );
      for (int ch = 0; ch < COUT; ++ch) row[c][ch] = o[ch];
    }
    pool.push(row, out);
  }
}

}  // namespace stream
