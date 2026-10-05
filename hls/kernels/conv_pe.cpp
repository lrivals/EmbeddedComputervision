// PE de convolution, interfaces et double tampon (T6.1, T6.2, T6.4 ; §10.2, §10.3 ; M10).
//
// Pour chaque tuile de sortie (row, col, to) :
//   conv_tile : pour ti = 0, Tn, … : load(ti + Tn) ∥ compute(ti)  — ping-pong in_buf / w_buf
//   store_tile(tuile précédente) ∥ conv_tile(tuile courante)      — ping-pong out_buf
// L'ordre des tuiles et leurs bornes sont ceux de `golden::conv_layer` pour la tuile
// (d.tr, d.tc) de la couche ; seule change la répartition des MAC (Tm × Tn par cycle, arbre
// d'additions sur Tn), ce qui ne change pas une somme d'entiers.
//
// Ports m_axi en mots de WORD octets (T10.1) : in_buf est chargé ligne par ligne, row_words(IC)
// mots par (voie, ligne) quel que soit l'alignement de la ligne en DDR ; w_buf par blocs
// contigus (poids réordonnés par le driver, weight_layout.hpp).
#include "accel.hpp"
#include "output_stage.hpp"

namespace accel {

#ifndef __SYNTHESIS__
SimCycles sim_cycles;
#endif

// load(in_buf) : n voies × (tr + K_MAX − 1) lignes × row_words(tc + K_MAX − 1) mots. Zéros hors
// de l'image (padding), hors de cin et au-delà de la tuile partielle ; voies ≥ n non écrites
// (masquées dans compute). Upsample et route par adressage (T6.4) : le canal `ch` est lu dans
// le segment 0 ou 1, à l'adresse divisée par 2^up. Conv pliée : la voie (c, i) contient la
// ligne + i du canal c.
// En matériel, in_buf est partitionné en WORD bancs sur les colonnes : un mot remplit jusqu'à
// WORD colonnes consécutives par cycle (2·WORD en upsample : II à confirmer en synthèse).
static void load_input(const word_t* act_in, const LayerDesc& d, const TileInfo& t, int ti,
                       int n, act_t in_buf[TN][IR][IC]) {
  const int K = d.k;
  const int rows = d.fold ? t.tr_n : t.tr_n + K - 1, cols = t.tc_n + K - 1;
  const int lr = d.tr + K_MAX - 1, lc = d.tc + K_MAX - 1, nw = row_words(lc);
  const int c0 = t.col - d.pad;  // première colonne logique de la tuile
  int tii = 0, rr = 0, k = 0;
load_in:
  for (int it = 0; it < n * lr * nw; ++it) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=IR max=TN*IR*3
    const int lane = ti + tii;
    const int ch = d.fold ? lane / K : lane, ki = d.fold ? lane % K : 0;
    const int ir = t.row - d.pad + rr + ki;
    const bool s1 = ch >= d.seg0_c;
    const int c = s1 ? ch - d.seg0_c : ch;
    const int off = s1 ? d.seg1_off : d.seg0_off;
    const int sh = s1 ? d.seg1_h : d.seg0_h, sw = s1 ? d.seg1_w : d.seg0_w;
    const int up = s1 ? d.seg1_up : d.seg0_up;
    const bool row_ok = ch < d.cin && rr < rows && ir >= 0 && ir < d.h;
    // Adresse d'octet du début de la ligne source, et premier mot lu.
    const int64_t base = off + (int64_t(c) * sh + (ir >> up)) * sw;
    const int64_t w0 = (base + ((c0 > 0 ? c0 : 0) >> up)) / WORD;
    const word_t v = row_ok ? act_in[w0 + k] : word_t(0);
    for (int cc = 0; cc < IC; ++cc) {
#pragma HLS UNROLL
      const int ic = c0 + cc;
      const bool col_ok = row_ok && cc < cols && ic >= 0 && ic < d.w;
      const int64_t q = base + (ic >> up) - (w0 + k) * WORD;  // octet dans le mot k
      if (col_ok && q >= 0 && q < WORD) in_buf[tii][rr][cc] = word_byte(v, int(q));
      else if (!col_ok && k == 0) in_buf[tii][rr][cc] = 0;
    }
    if (++k == nw) {
      k = 0;
      if (++rr == lr) {
        rr = 0;
        ++tii;
      }
    }
  }
}

// load(w_buf) : bloc contigu de Tm × n × kh × k octets, ordre (i, j, tii, too).
static void load_weights(const word_t* wts, const LayerDesc& d, const TileInfo& t, int kti,
                         int n, w_t w_buf[TM][TN][K_MAX][K_MAX]) {
  const int K = d.k;
  const int64_t bytes = w_block_bytes(d, n);
  const int64_t w0 = (d.w_off + w_block_off(d, t.to / TM, kti)) / WORD;
load_w:
  for (int k = 0; k < cdiv(int(bytes), WORD); ++k) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TM*TN/WORD max=TM*TN*K_MAX*K_MAX/WORD
    const word_t v = wts[w0 + k];
    for (int b = 0; b < WORD; ++b) {
#pragma HLS UNROLL
      const int e = k * WORD + b;
      if (e < bytes) {
        const int too = e % TM, tii = (e / TM) % n, ij = e / (TM * n);
        w_buf[too][tii][ij / K][ij % K] = word_byte(v, b);
      }
    }
  }
}

// Tm × Tn MAC par cycle : boucle (i, j, trr, tcc) aplatie en PIPELINE II=1, too et tii
// déroulés, arbre d'additions sur Tn ; voies ≥ n masquées (trim). `first` : première ti,
// out_buf repart de zéro.
static void compute(const act_t in_buf[TN][IR][IC], const w_t w_buf[TM][TN][K_MAX][K_MAX],
                    acc_t out_buf[TM][TRB][TCB], int kh, int kw, int tr, int tc, int n,
                    bool first) {
  int i = 0, j = 0, trr = 0, tcc = 0;
mac:
  for (int it = 0; it < kh * kw * tr * tc; ++it) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TR*TC max=K_MAX*K_MAX*TRB*TCB
#pragma HLS DEPENDENCE variable=out_buf inter false
    const bool init = first && i == 0 && j == 0;
    for (int too = 0; too < TM; ++too) {
#pragma HLS UNROLL
      acc_t sum = 0;
      for (int tii = 0; tii < TN; ++tii) {
#pragma HLS UNROLL
        if (tii < n) sum += mul_w(int(w_buf[too][tii][i][j]), int(in_buf[tii][trr + i][tcc + j]));
      }
      out_buf[too][trr][tcc] = (init ? acc_t(0) : out_buf[too][trr][tcc]) + sum;
    }
    if (++tcc == tc) {
      tcc = 0;
      if (++trr == tr) {
        trr = 0;
        if (++j == kw) {
          j = 0;
          ++i;
        }
      }
    }
  }
}

// Une tuile de sortie complète : accumulation sur ti avec double tampon in_buf / w_buf.
static void conv_tile(const word_t* act_in, const word_t* wts, const LayerDesc& d,
                      const TileInfo& t, act_t in_a[TN][IR][IC], act_t in_b[TN][IR][IC],
                      w_t w_a[TM][TN][K_MAX][K_MAX], w_t w_b[TM][TN][K_MAX][K_MAX],
                      acc_t out_buf[TM][TRB][TCB]) {
  const int nti = n_ti(d), kh = kern_h(d);
  load_input(act_in, d, t, 0, ti_lanes(d, 0), in_a);
  load_weights(wts, d, t, 0, ti_lanes(d, 0), w_a);
ti_loop:
  for (int k = 0; k < nti; ++k) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=43
    const bool more = k + 1 < nti;
    const int n = ti_lanes(d, k), n_next = more ? ti_lanes(d, k + 1) : 0;
    if (k % 2 == 0) {
      if (more) {
        load_input(act_in, d, t, (k + 1) * TN, n_next, in_b);
        load_weights(wts, d, t, k + 1, n_next, w_b);
      }
      compute(in_a, w_a, out_buf, kh, d.k, d.tr, d.tc, n, k == 0);
    } else {
      if (more) {
        load_input(act_in, d, t, (k + 1) * TN, n_next, in_a);
        load_weights(wts, d, t, k + 1, n_next, w_a);
      }
      compute(in_b, w_b, out_buf, kh, d.k, d.tr, d.tc, n, k == 0);
    }
  }
}

#ifndef __SYNTHESIS__
// Même formule que tools/perf_model.py (layer_cycles).
static void count_cycles(const LayerDesc& d, const TileInfo& t, bool first_tile,
                         uint64_t& prev_store) {
  const int nti = n_ti(d);
  const uint64_t lr = uint64_t(d.tr) + K_MAX - 1, nw = row_words(d.tc + K_MAX - 1);
  const uint64_t comp = uint64_t(kern_h(d)) * d.k * d.tr * d.tc;
  const int pk = d.pk(), ps = d.ps();
  const int Pr = (d.tr - pk) / ps + 1, Pc = (d.tc - pk) / ps + 1;
  const uint64_t own = d.pooled() && d.prepool_off >= 0
                           ? uint64_t(imin(t.tr_n, Pr * ps)) * row_words(imin(t.tc_n, Pc * ps))
                           : 0;
  const uint64_t st = uint64_t(cdiv(t.tm_n, RQ)) * d.tr * d.tc +
                      uint64_t(t.tm_n) * (own + uint64_t(t.np_r) * row_words(t.np_c));

  SimCycles& s = sim_cycles;
  uint64_t tile = 0;
  for (int k = 0; k < nti; ++k) {
    const int n = ti_lanes(d, k);
    const uint64_t ld_in = uint64_t(n) * lr * nw;
    const uint64_t ld_w = uint64_t(cdiv(int(w_block_bytes(d, n)), WORD));
    const uint64_t ld = ld_in > ld_w ? ld_in : ld_w;  // deux bundles m_axi : en parallèle
    s.load_in += ld_in;
    s.load_w += ld_w;
    s.compute += comp;
    s.sequential += ld_in + ld_w + comp;
    // Ping-pong : chargement de ti = 0, puis max(chargement ti + 1, calcul ti).
    tile += k == 0 ? ld : (ld > comp ? ld : comp);
  }
  tile += comp;
  s.store += st;
  s.sequential += st;
  s.overlapped += first_tile ? tile : (tile > prev_store ? tile : prev_store);
  prev_store = st;
}
#endif

// Une couche conv (+ maxpool fusionné), décrite par `d`.
static void conv_layer(const word_t* act_in, word_t* act_out, const word_t* wts,
                       const int32_t* prm, const LayerDesc& d, act_t in_a[TN][IR][IC],
                       act_t in_b[TN][IR][IC], w_t w_a[TM][TN][K_MAX][K_MAX],
                       w_t w_b[TM][TN][K_MAX][K_MAX], acc_t out_a[TM][TRB][TCB],
                       acc_t out_b[TM][TRB][TCB]) {
  const int R = d.out_h(), C = d.out_w();
  const int pk = d.pk(), ps = d.ps();
  const int Rp = d.pool_h(), Cp = d.pool_w();
  const int Pr = (d.tr - pk) / ps + 1, Pc = (d.tc - pk) / ps + 1;  // lignes poolées / tuile
  const int n_r = (Rp + Pr - 1) / Pr, n_c = (Cp + Pc - 1) / Pc, n_m = (d.cout + TM - 1) / TM;
  const int n_tiles = n_r * n_c * n_m;

#ifndef __SYNTHESIS__
  uint64_t prev_store = 0;
#endif

  // Parcours row → col → to (golden) ; le stockage de la tuile k − 1 se fait pendant le
  // calcul de la tuile k, dans l'autre out_buf.
  TileInfo prev{};
  int ir = 0, ic = 0, im = 0;
tiles:
  for (int k = 0; k <= n_tiles; ++k) {
#pragma HLS LOOP_TRIPCOUNT min=2 max=4097
    TileInfo cur;
    cur.prow0 = ir * Pr;
    cur.pcol0 = ic * Pc;
    cur.row = cur.prow0 * ps;
    cur.col = cur.pcol0 * ps;
    cur.to = im * TM;
    cur.tr_n = imin((Pr - 1) * ps + pk, R - cur.row);
    cur.tc_n = imin((Pc - 1) * ps + pk, C - cur.col);
    cur.np_r = imin(Pr, Rp - cur.prow0);
    cur.np_c = imin(Pc, Cp - cur.pcol0);
    cur.tm_n = imin(TM, d.cout - cur.to);

    const bool do_conv = k < n_tiles, do_store = k > 0;
    if (k % 2 == 0) {
      if (do_conv) conv_tile(act_in, wts, d, cur, in_a, in_b, w_a, w_b, out_a);
      if (do_store) store_tile(act_out, prm, d, prev, out_b);
    } else {
      if (do_conv) conv_tile(act_in, wts, d, cur, in_a, in_b, w_a, w_b, out_b);
      if (do_store) store_tile(act_out, prm, d, prev, out_a);
    }
#ifndef __SYNTHESIS__
    if (do_conv) count_cycles(d, cur, k == 0, prev_store);
    else sim_cycles.overlapped += prev_store;
#endif
    prev = cur;
    if (++im == n_m) {
      im = 0;
      if (++ic == n_c) {
        ic = 0;
        ++ir;
      }
    }
  }
}

}  // namespace accel

using namespace accel;

void yolo_conv(const word_t* act_in, word_t* act_out, const word_t* wts, const int32_t* prm,
               LayerDesc d, const int32_t* descs, int32_t n_calls) {
#pragma HLS INTERFACE m_axi port=act_in offset=slave bundle=gmem_in depth=131072
#pragma HLS INTERFACE m_axi port=act_out offset=slave bundle=gmem_out depth=131072
#pragma HLS INTERFACE m_axi port=wts offset=slave bundle=gmem_w depth=1105688
#pragma HLS INTERFACE m_axi port=prm offset=slave bundle=gmem_p depth=7392
#pragma HLS INTERFACE m_axi port=descs offset=slave bundle=gmem_p depth=720
#pragma HLS INTERFACE s_axilite port=act_in bundle=control
#pragma HLS INTERFACE s_axilite port=act_out bundle=control
#pragma HLS INTERFACE s_axilite port=wts bundle=control
#pragma HLS INTERFACE s_axilite port=prm bundle=control
#pragma HLS INTERFACE s_axilite port=d bundle=control
#pragma HLS INTERFACE s_axilite port=descs bundle=control
#pragma HLS INTERFACE s_axilite port=n_calls bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS AGGREGATE variable=d

  // Tampons sur puce en double (ping-pong) — 2 × (B_in + B_w + B_out), §10.2. in_buf en WORD
  // bancs sur les colonnes (un mot par cycle, T10.1) ; w_buf en registres sur (too, tii).
  static act_t in_a[TN][IR][IC], in_b[TN][IR][IC];
  static w_t w_a[TM][TN][K_MAX][K_MAX], w_b[TM][TN][K_MAX][K_MAX];
  static acc_t out_a[TM][TRB][TCB], out_b[TM][TRB][TCB];
#pragma HLS ARRAY_PARTITION variable=in_a complete dim=1
#pragma HLS ARRAY_PARTITION variable=in_b complete dim=1
#pragma HLS ARRAY_PARTITION variable=in_a cyclic factor=ACC_WORD_BYTES dim=3
#pragma HLS ARRAY_PARTITION variable=in_b cyclic factor=ACC_WORD_BYTES dim=3
#pragma HLS ARRAY_PARTITION variable=w_a complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_a complete dim=2
#pragma HLS ARRAY_PARTITION variable=w_b complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_b complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_a complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_b complete dim=1

#ifndef __SYNTHESIS__
  sim_cycles = SimCycles{};
#endif
  if (n_calls == 0) {
    conv_layer(act_in, act_out, wts, prm, d, in_a, in_b, w_a, w_b, out_a, out_b);
    return;
  }
  // Séquenceur (T10.7) : descripteurs lus en DDR, couches enchaînées sans l'ARM.
seq:
  for (int c = 0; c < n_calls; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=24
    int32_t w[D_WORDS];
    for (int i = 0; i < D_WORDS; ++i) {
#pragma HLS PIPELINE II=1
      w[i] = descs[c * D_WORDS + i];
    }
    conv_layer(act_in, act_out, wts, prm, desc_from_words(w), in_a, in_b, w_a, w_b, out_a,
               out_b);
  }
}
