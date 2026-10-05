// Étage streaming d'une conv (T9.4.2, §10.1) : un bloc matériel par couche
// [2018-venieris#004.0], activations en flux HWC (pixel par pixel, canaux contigus).
//
// - Fenêtre glissante par line buffer [2023-montgomerie-corcoran#011.0] : K lignes
//   circulaires de W·C_in mots (les K − 1 lignes du §10.2 et la ligne en cours d'arrivée) ;
//   la ligne de sortie r est calculée dès que la ligne d'entrée r + pad est arrivée.
// - Mode `FRAME` (poids en DDR, hybride de tools/stream_model.py) : toute la carte d'entrée
//   est gardée sur la puce et la boucle des groupes PE est à l'extérieur (T10.8) : les poids
//   d'un groupe (PE × C_in × K² octets) sont lus une fois en DDR dans un tampon local, puis
//   toute la carte est balayée ; chaque poids est donc lu une fois par image. Sortie : canal
//   par canal (CHW, `OUT_CHW`, PE = 1) vers un étage FRAME suivant, qui la range telle quelle
//   (`IN_CHW`), ou carte de sortie émise en HWC après le dernier groupe.
// - Poids des étages sur la puce : ROM générée par tools/gen_stream_rom.py (T10.8), rangée
//   [og][ig][i][j][pe·SIMD + s] pour qu'un mot de PE·SIMD octets soit lu par cycle ; sans ROM
//   (C-sim de référence), lecture directe de weights.bin (C_out, C_in, K, K).
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
  const int8_t* w = nullptr;      // (C_out, C_in, K, K) dans weights.bin (DDR)
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

// Accumulation d'un groupe de PE canaux de sortie (og) pour un pixel : poids `wt(o, c, i, j)`,
// fenêtre `px(c, i, j)`. Une itération (ig, i, j) = un cycle (II = 1).
template <int K, int CIN, int PE, int SIMD, class Wt, class Px>
inline void conv_group(Wt wt, Px px, int og, int32_t acc[PE]
#ifndef __SYNTHESIS__
                       , StageCycles& cyc
#endif
) {
  static_assert(CIN % SIMD == 0, "repliement : SIMD | C_in");
  for (int pe = 0; pe < PE; ++pe) acc[pe] = 0;
  for (int ig = 0; ig < CIN; ig += SIMD)
    for (int i = 0; i < K; ++i)
      for (int j = 0; j < K; ++j) {
#pragma HLS PIPELINE II=1
#ifndef __SYNTHESIS__
        ++cyc.mac;
#endif
        for (int pe = 0; pe < PE; ++pe)
          for (int s = 0; s < SIMD; ++s)
            acc[pe] += int32_t(wt(og + pe, ig + s, i, j)) * px(ig + s, i, j);
      }
}

// Étage de sortie d'un canal : arithmétique du golden.
inline int8_t requant_out(const StageParams& p, int o, int32_t acc) {
  int32_t y = golden::requantize(acc + p.bias[o], p.m0[o], p.shift);
  if (p.leaky) y = golden::leaky_int(y);
  return golden::clip_q(y, p.qmax);
}

// Poids d'une ROM [og][ig][i][j][pe·SIMD + s] (gen_stream_rom.py).
template <int K, int CIN, int PE, int SIMD>
inline int8_t rom_at(const int8_t* rom, int o, int c, int i, int j) {
  return rom[((((o / PE) * (CIN / SIMD) + c / SIMD) * K + i) * K + j) * (PE * SIMD) +
             (o % PE) * SIMD + c % SIMD];
}

template <int K, int CIN, int COUT, int H, int W, int PE, int SIMD, int PK, int PS,
          bool FRAME, const int8_t* ROM = nullptr, bool IN_CHW = false, bool OUT_CHW = false>
void conv_stage(hls::stream<int8_t>& in, hls::stream<int8_t>& out, const StageParams& p
#ifndef __SYNTHESIS__
                , StageCycles& cyc
#endif
) {
  static_assert(COUT % PE == 0, "repliement : PE | C_out");
  static_assert(FRAME || (!IN_CHW && !OUT_CHW), "CHW : étages FRAME seulement");
  static_assert(!OUT_CHW || (PE == 1 && PK == 0), "sortie CHW : PE = 1, sans maxpool");
  static_assert(!(FRAME && ROM), "étage FRAME : poids en DDR");
  constexpr int PAD = K / 2;
  constexpr int ROWS = FRAME ? H : K;  // carte entière ou line buffer circulaire
  static int8_t buf[ROWS][W][CIN];
  static int8_t row[W][COUT];
  static PoolUnit<H, W, COUT, PK, PS> pool;
  pool.rows = 0;

  if constexpr (FRAME) {
    // Carte d'entrée entière (HWC, ou CHW depuis un étage FRAME).
    if constexpr (IN_CHW) {
      for (int ch = 0; ch < CIN; ++ch)
        for (int r = 0; r < H; ++r)
          for (int c = 0; c < W; ++c) buf[r][c][ch] = in.read();
    } else {
      for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
          for (int ch = 0; ch < CIN; ++ch) buf[r][c][ch] = in.read();
    }
    static int8_t ofm[OUT_CHW ? 1 : H][OUT_CHW ? 1 : W][COUT];
    static int8_t wloc[PE][CIN][K][K];  // poids du groupe, lus une fois en DDR
    for (int og = 0; og < COUT; og += PE) {
      for (int pe = 0; pe < PE; ++pe)
        for (int c = 0; c < CIN; ++c)
          for (int i = 0; i < K; ++i)
            for (int j = 0; j < K; ++j)
              wloc[pe][c][i][j] = p.w[(((og + pe) * CIN + c) * K + i) * K + j];
      auto wt = [&](int o, int c, int i, int j) -> int8_t { return wloc[o - og][c][i][j]; };
      for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c) {
          auto px = [&](int ch, int i, int j) -> int32_t {
            const int ir = r - PAD + i, ic = c - PAD + j;
            if (ir < 0 || ir >= H || ic < 0 || ic >= W) return 0;
            return buf[ir][ic][ch];
          };
          int32_t acc[PE];
          conv_group<K, CIN, PE, SIMD>(wt, px, og, acc
#ifndef __SYNTHESIS__
                                       , cyc
#endif
          );
          for (int pe = 0; pe < PE; ++pe) {
            const int8_t y = requant_out(p, og + pe, acc[pe]);
            if constexpr (OUT_CHW) out.write(y);
            else ofm[r][c][og + pe] = y;
          }
        }
    }
    if constexpr (!OUT_CHW)
      for (int r = 0; r < H; ++r) pool.push(ofm[r], out);
  } else {
    // Line buffer : la ligne de sortie r est calculée dès que la ligne r + pad est arrivée.
    auto wt = [&](int o, int c, int i, int j) -> int8_t {
      if constexpr (ROM != nullptr) return rom_at<K, CIN, PE, SIMD>(ROM, o, c, i, j);
      else return p.w[((o * CIN + c) * K + i) * K + j];
    };
    int read = 0;
    auto read_row = [&]() {
      for (int c = 0; c < W; ++c)
        for (int ch = 0; ch < CIN; ++ch) buf[read % ROWS][c][ch] = in.read();
      ++read;
    };
    for (int r = 0; r < H; ++r) {
      while (read <= (r + PAD < H ? r + PAD : H - 1)) read_row();
      for (int c = 0; c < W; ++c) {
        auto px = [&](int ch, int i, int j) -> int32_t {
          const int ir = r - PAD + i, ic = c - PAD + j;
          if (ir < 0 || ir >= H || ic < 0 || ic >= W) return 0;  // complétion par des zéros
          return buf[ir % ROWS][ic][ch];
        };
        for (int og = 0; og < COUT; og += PE) {
          int32_t acc[PE];
          conv_group<K, CIN, PE, SIMD>(wt, px, og, acc
  #ifndef __SYNTHESIS__
                                       , cyc
  #endif
          );
          for (int pe = 0; pe < PE; ++pe) row[c][og + pe] = requant_out(p, og + pe, acc[pe]);
        }
      }
      pool.push(row, out);
    }
  }
}

}  // namespace stream
