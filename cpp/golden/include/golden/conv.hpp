// Convolution tuilée du moteur unique, comme le matériel l'exécute — §10.2, §9.3, §10.3.
//
// En-tête seul, sans allocation dynamique ni bibliothèque non synthétisable (ADR 0002) : les
// tampons sur puce `in_buf`, `w_buf`, `out_buf` ont une taille fixée par les tuiles
// (Tm, Tn, Tr, Tc), paramètres de template.
//
// Écarts au pseudo-code du §10.2 (voir docs/tasks/M5-golden-cpp.md, T5.2) :
// - ordre des boucles de tuiles row → col → to → ti (2015-zhang) : `out_buf` accumule sur ti
//   et part en DDR après la dernière ti ; l'ordre (to, ti, row, col) du §10.2 demanderait un
//   `out_buf` de la taille de la carte entière ;
// - `Tr`, `Tc` sont la tuile de sortie de la conv **avant** pooling. Une conv à maxpool
//   fusionné (k_p, s_p) produit P = ⌊(Tr − k_p)/s_p⌋ + 1 lignes poolées par tuile et avance
//   de P·s_p lignes conv ; en stride 1 la dernière ligne de la tuile est recalculée par la
//   suivante, et le bord bas/droit est répliqué (§10.3, conventions.md). Sans pooling, on
//   prend k_p = s_p = 1 (P = Tr).
// - dans une tuile, l'ordre des MAC (too, tii, i, j, trr, tcc) diffère du §10.2 par simple
//   commutation : la somme d'entiers est exacte, le résultat est le même que l'arbre
//   d'additions déroulé du matériel.
#pragma once

#include <cstdint>

namespace golden {

constexpr int K_MAX = 3;   // plus grand noyau des réseaux Tiny
constexpr int S_CONV = 1;  // toutes les convs des réseaux Tiny sont de stride 1
constexpr int MAX_SEGMENTS = 2;

// §10.2 : B_in = Tn (S Tr + K − S)(S Tc + K − S), B_w = Tm Tn K², B_out = Tm Tr Tc (éléments).
constexpr int64_t buf_in_size(int Tn, int Tr, int Tc, int K, int S) {
  return int64_t(Tn) * (S * Tr + K - S) * (S * Tc + K - S);
}
constexpr int64_t buf_w_size(int Tm, int Tn, int K) { return int64_t(Tm) * Tn * K * K; }
constexpr int64_t buf_out_size(int Tm, int Tr, int Tc) { return int64_t(Tm) * Tr * Tc; }

// Morceau de l'entrée d'une conv : `c` canaux (h, w) int8 contigus à `base`, lus à travers
// une division d'adresse par 2^up (upsample ×2 sans copie, §10.3).
struct Segment {
  const int8_t* base = nullptr;
  int c = 0, h = 0, w = 0;  // forme rangée en mémoire
  int up = 0;
};

// Entrée logique d'une conv : concaténation sur les canaux de segments (route, §10.3).
struct InView {
  Segment seg[MAX_SEGMENTS];
  int nseg = 0;
  int c = 0, h = 0, w = 0;  // forme logique

  int8_t at(int ch, int r, int col) const {
    for (int s = 0; s < nseg; ++s) {
      const Segment& g = seg[s];
      if (ch < g.c)
        return g.base[(int64_t(ch) * g.h + (r >> g.up)) * g.w + (col >> g.up)];
      ch -= g.c;
    }
    return 0;  // hors des canaux : jamais lu (le chargeur borne les canaux)
  }
};

inline InView single_view(const int8_t* base, int c, int h, int w) {
  InView v;
  v.seg[0] = Segment{base, c, h, w, 0};
  v.nseg = 1;
  v.c = c;
  v.h = h;
  v.w = w;
  return v;
}

struct ConvParams {
  int k = 3, pad = 1;
  int cin = 0, cout = 0;
  int h = 0, w = 0;      // entrée
  int shift = 31;        // n de la requantification
  int qmax = 127;        // saturation de la sortie (T9.3)
  bool leaky = true;
  int pool_k = 0, pool_s = 0;  // maxpool fusionné ; 0 : aucun
  // ABFT (T25.2): checksum filter Σ_f w_f (cin, k, k) and Σ_f b_f; nullptr: no check.
  const int32_t* cs_w = nullptr;
  int32_t cs_b = 0;

  int out_h() const { return h + 2 * pad - k + 1; }  // stride 1
  int out_w() const { return w + 2 * pad - k + 1; }
  bool pooled() const { return pool_k > 0; }
  int pk() const { return pooled() ? pool_k : 1; }
  int ps() const { return pooled() ? pool_s : 1; }
  // §4.4 : stride 1 → complétion par réplication, la taille est conservée.
  int pool_h() const { return ps() == 1 ? out_h() : (out_h() - pk()) / ps() + 1; }
  int pool_w() const { return ps() == 1 ? out_w() : (out_w() - pk()) / ps() + 1; }
};

// Octets échangés avec la DDR (compteurs d'accès, T5.3).
struct AccessStats {
  uint64_t act_rd = 0;    // activations lues
  uint64_t act_wr = 0;    // activations écrites
  uint64_t param_rd = 0;  // poids, biais, M0
  // ABFT (T25.2): output pixels checked and pixels whose checksum failed.
  uint64_t abft_checks = 0;
  uint64_t abft_errors = 0;
};

// Contrat arithmétique (conventions.md « Arithmétique entière », int_layers.py).
inline int32_t requantize(int32_t acc, int32_t m0, int n) {
  // §9.3 : (acc · M0 + 2ⁿ⁻¹) >> n sur 64 bits, décalage arithmétique.
  return int32_t((int64_t(acc) * m0 + (int64_t(1) << (n - 1))) >> n);
}
inline int32_t leaky_int(int32_t y) { return y > 0 ? y : (13 * y + 64) >> 7; }
inline int8_t clip_q(int32_t y, int qmax) {
  return int8_t(y < -qmax ? -qmax : (y > qmax ? qmax : y));
}
inline int8_t clip8(int32_t y) { return clip_q(y, 127); }

// T9.3.4 [2024-yan#007.3] : produit d'une entrée 8 bits par deux multiplieurs à entrée 4 bits,
// x = 16·(x ≫ 4) + (x & 15) ; égal à w·x pour tout (w, x) int8.
inline int32_t mul_split4(int8_t w, int8_t x) {
  return 16 * (int32_t(w) * (int32_t(x) >> 4)) + int32_t(w) * (int32_t(x) & 15);
}

// T23 [2024-zhang#011.0] : deux produits int8 qui partagent x dans un multiplieur DSP48E2
// (A 27 bits × B 18 bits, P 48 bits), python/yolo/quant/pack.py. (w_a·2^S + w_b)·x =
// (w_a·x)·2^S + w_b·x ; une somme de PACK_N produits paquetés reste exacte tant que le canal
// bas tient sur S bits signés : PACK_N = 8 sous le contrat (|w|, |x| ≤ 127), 7 pour tout int8.
constexpr int PACK_S = 18, PACK_N = 8;
inline int64_t pack2(int8_t wa, int8_t wb) { return int64_t(wa) * (int64_t(1) << PACK_S) + wb; }
inline int64_t mac2(int8_t wa, int8_t wb, int8_t x) { return pack2(wa, wb) * x; }
// Canal bas étendu en signe ; s'il est négatif, il a emprunté 1 au canal haut.
inline void unpack2(int64_t p, int32_t& hi, int32_t& lo) {
  constexpr int64_t half = int64_t(1) << (PACK_S - 1);
  lo = int32_t(((p & ((int64_t(1) << PACK_S) - 1)) ^ half) - half);
  hi = int32_t((p >> PACK_S) + (lo < 0));
}

// Tampons sur puce d'une configuration de tuiles (taille == formules du §10.2).
template <int TM, int TN, int TR, int TC>
struct Tiles {
  static constexpr int Tm = TM, Tn = TN, Tr = TR, Tc = TC;
  static constexpr int IR = S_CONV * TR + K_MAX - S_CONV;
  static constexpr int IC = S_CONV * TC + K_MAX - S_CONV;

  struct Buffers {
    int8_t in_buf[TN][IR][IC];
    int8_t w_buf[TM][TN][K_MAX][K_MAX];
    int32_t out_buf[TM][TR][TC];
    // ABFT (T25.2): checksum lane conv(x, Σ_f w_f) and Σ_f (acc_f + b_f) per pixel of the
    // spatial tile, modulo 2³² (unsigned: wrap without undefined behaviour).
    uint32_t cs_buf[TR][TC];
    uint32_t sum_buf[TR][TC];
  };
  static_assert(sizeof(Buffers::in_buf) == buf_in_size(TN, TR, TC, K_MAX, S_CONV), "B_in");
  static_assert(sizeof(Buffers::w_buf) == buf_w_size(TM, TN, K_MAX), "B_w");
  static_assert(sizeof(Buffers::out_buf) == buf_out_size(TM, TR, TC) * sizeof(int32_t), "B_out");
  static_assert(TR >= 2 && TC >= 2, "une tuile doit contenir une fenêtre de maxpool 2×2");
};

namespace detail {
inline int imin(int a, int b) { return a < b ? a : b; }
}

// Une couche conv complète : nid de boucles tuilé, puis étage de sortie biais +
// requantification + leaky + saturation (+ maxpool fusionné).
// - `w` : (cout, cin, k, k) int8 ; `bias`, `m0` : un int32 par canal de sortie ;
// - `out` : (cout, pool_h, pool_w) — la carte poolée si `p.pooled()` ;
// - `prepool` : carte avant pooling en DDR (`prepool_out`, route 20) ou nullptr ;
// - `tap` : même carte, sonde de débogage hors DDR (non comptée) ou nullptr.
template <class T>
void conv_layer(const ConvParams& p, const InView& in, const int8_t* w, const int32_t* bias,
                const int32_t* m0, int8_t* out, int8_t* prepool, int8_t* tap,
                AccessStats* stats, typename T::Buffers& b) {
  using detail::imin;
  const int K = p.k, S = S_CONV;
  const int R = p.out_h(), C = p.out_w();
  const int pk = p.pk(), ps = p.ps();
  const int Rp = p.pool_h(), Cp = p.pool_w();
  const int Pr = (T::Tr - pk) / ps + 1, Pc = (T::Tc - pk) / ps + 1;  // lignes poolées / tuile
  AccessStats st;

  for (int prow0 = 0; prow0 < Rp; prow0 += Pr) {
    const int row = prow0 * ps;
    const int tr_n = imin((Pr - 1) * ps + pk, R - row);  // lignes conv calculées
    const int np_r = imin(Pr, Rp - prow0);               // lignes poolées produites
    for (int pcol0 = 0; pcol0 < Cp; pcol0 += Pc) {
      const int col = pcol0 * ps;
      const int tc_n = imin((Pc - 1) * ps + pk, C - col);
      const int np_c = imin(Pc, Cp - pcol0);
      // ABFT (T25.2): Σ over every channel tile `to` of the int32 accumulators + bias,
      // against the checksum lane computed once (to == 0) from the same in_buf.
      const bool abft = p.cs_w != nullptr;
      if (abft)
        for (int trr = 0; trr < tr_n; ++trr)
          for (int tcc = 0; tcc < tc_n; ++tcc) b.cs_buf[trr][tcc] = b.sum_buf[trr][tcc] = 0;
      for (int to = 0; to < p.cout; to += T::Tm) {
        const int tm_n = imin(T::Tm, p.cout - to);
        for (int ti = 0; ti < p.cin; ti += T::Tn) {
          // load(in_buf) : zéros hors de l'image (padding) et au-delà de cin.
          for (int tii = 0; tii < T::Tn; ++tii)
            for (int rr = 0; rr < T::IR; ++rr)
              for (int cc = 0; cc < T::IC; ++cc) {
                const int ch = ti + tii, ir = row * S - p.pad + rr, ic = col * S - p.pad + cc;
                int8_t v = 0;
                if (ch < p.cin && rr < S * tr_n + K - S && cc < S * tc_n + K - S && ir >= 0 &&
                    ir < p.h && ic >= 0 && ic < p.w) {
                  v = in.at(ch, ir, ic);
                  ++st.act_rd;
                }
                b.in_buf[tii][rr][cc] = v;
              }
          // load(w_buf)
          for (int too = 0; too < T::Tm; ++too)
            for (int tii = 0; tii < T::Tn; ++tii)
              for (int i = 0; i < K_MAX; ++i)
                for (int j = 0; j < K_MAX; ++j) {
                  const int o = to + too, c = ti + tii;
                  int8_t v = 0;
                  if (o < p.cout && c < p.cin && i < K && j < K) {
                    v = w[((int64_t(o) * p.cin + c) * K + i) * K + j];
                    ++st.param_rd;
                  }
                  b.w_buf[too][tii][i][j] = v;
                }
          if (ti == 0)
            for (int too = 0; too < T::Tm; ++too)
              for (int trr = 0; trr < T::Tr; ++trr)
                for (int tcc = 0; tcc < T::Tc; ++tcc) b.out_buf[too][trr][tcc] = 0;
          // Calcul : Tm × Tn MAC par cycle dans le matériel (§10.2).
          for (int too = 0; too < tm_n; ++too)
            for (int tii = 0; tii < T::Tn; ++tii)
              for (int i = 0; i < K; ++i)
                for (int j = 0; j < K; ++j) {
                  const int32_t wv = b.w_buf[too][tii][i][j];
                  for (int trr = 0; trr < tr_n; ++trr) {
                    int32_t* o = b.out_buf[too][trr];
                    const int8_t* x = &b.in_buf[tii][S * trr + i][j];
                    for (int tcc = 0; tcc < tc_n; ++tcc) o[tcc] += wv * x[S * tcc];
                  }
                }
          // Checksum lane (one more output lane in the hardware, T25.3).
          if (abft && to == 0)
            for (int tii = 0; tii < T::Tn && ti + tii < p.cin; ++tii)
              for (int i = 0; i < K; ++i)
                for (int j = 0; j < K; ++j) {
                  const uint32_t wv = uint32_t(p.cs_w[((ti + tii) * K + i) * K + j]);
                  for (int trr = 0; trr < tr_n; ++trr) {
                    const int8_t* x = &b.in_buf[tii][S * trr + i][j];
                    for (int tcc = 0; tcc < tc_n; ++tcc)
                      b.cs_buf[trr][tcc] += wv * uint32_t(int32_t(x[S * tcc]));
                  }
                }
        }
        // store(out_buf) : biais + requantification + leaky + saturation (§9.3).
        st.param_rd += 8 * uint64_t(tm_n);
        for (int too = 0; too < tm_n; ++too) {
          const int o = to + too;
          for (int trr = 0; trr < tr_n; ++trr)
            for (int tcc = 0; tcc < tc_n; ++tcc) {
              if (abft) b.sum_buf[trr][tcc] += uint32_t(b.out_buf[too][trr][tcc]) + uint32_t(bias[o]);
              int32_t y = requantize(b.out_buf[too][trr][tcc] + bias[o], m0[o], p.shift);
              if (p.leaky) y = leaky_int(y);
              b.out_buf[too][trr][tcc] = clip_q(y, p.qmax);
            }
          // Carte avant pooling : lignes et colonnes propres à la tuile (hors recouvrement).
          const int own_r = imin(tr_n, Pr * ps), own_c = imin(tc_n, Pc * ps);
          if (p.pooled() && (prepool || tap))
            for (int trr = 0; trr < own_r; ++trr)
              for (int tcc = 0; tcc < own_c; ++tcc) {
                const int64_t a = (int64_t(o) * R + row + trr) * C + col + tcc;
                const int8_t v = int8_t(b.out_buf[too][trr][tcc]);
                if (prepool) {
                  prepool[a] = v;
                  ++st.act_wr;
                }
                if (tap) tap[a] = v;
              }
          // Maxpool fusionné (§10.3) ; fenêtre bornée à la carte : réplication du bord.
          for (int pr = 0; pr < np_r; ++pr)
            for (int pc = 0; pc < np_c; ++pc) {
              int32_t m = -128;
              for (int a = 0; a < pk; ++a)
                for (int c = 0; c < pk; ++c) {
                  const int r = imin((prow0 + pr) * ps + a, R - 1) - row;
                  const int q = imin((pcol0 + pc) * ps + c, C - 1) - col;
                  const int32_t v = b.out_buf[too][r][q];
                  m = v > m ? v : m;
                }
              out[(int64_t(o) * Rp + prow0 + pr) * Cp + pcol0 + pc] = int8_t(m);
              ++st.act_wr;
            }
        }
      }
      if (abft) {
        for (int trr = 0; trr < tr_n; ++trr)
          for (int tcc = 0; tcc < tc_n; ++tcc)
            st.abft_errors += b.cs_buf[trr][tcc] + uint32_t(p.cs_b) != b.sum_buf[trr][tcc];
        st.abft_checks += uint64_t(tr_n) * tc_n;
      }
    }
  }
  if (stats) {
    stats->act_rd += st.act_rd;
    stats->act_wr += st.act_wr;
    stats->param_rd += st.param_rd;
    stats->abft_checks += st.abft_checks;
    stats->abft_errors += st.abft_errors;
  }
}

// Variante qui porte ses propres tampons (pile) — usage normal du golden.
template <class T>
void conv_layer(const ConvParams& p, const InView& in, const int8_t* w, const int32_t* bias,
                const int32_t* m0, int8_t* out, int8_t* prepool = nullptr,
                int8_t* tap = nullptr, AccessStats* stats = nullptr) {
  static thread_local typename T::Buffers b;  // taille fixe, réutilisée d'un appel à l'autre
  conv_layer<T>(p, in, w, bias, m0, out, prepool, tap, stats, b);
}

}  // namespace golden
