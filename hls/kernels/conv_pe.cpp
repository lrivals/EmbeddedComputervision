// PE de convolution, interfaces et double tampon (T6.1, T6.2, T6.4 ; §10.2, §10.3).
//
// Pour chaque tuile de sortie (row, col, to) :
//   conv_tile : pour ti = 0, Tn, … : load(ti + Tn) ∥ compute(ti)  — ping-pong in_buf / w_buf
//   store_tile(tuile précédente) ∥ conv_tile(tuile courante)      — ping-pong out_buf
// L'ordre des tuiles, leurs bornes et le contenu des tampons sont ceux de
// `golden::conv_layer` ; seule change la répartition des MAC (Tm × Tn par cycle, arbre
// d'additions sur Tn), ce qui ne change pas une somme d'entiers.
#include "accel.hpp"
#include "output_stage.hpp"

namespace accel {

#ifndef __SYNTHESIS__
SimCycles sim_cycles;
#endif

// load(in_buf) : zéros hors de l'image (padding), au-delà de cin et de la tuile partielle.
// Upsample et route par adressage (T6.4) : le canal `ch` est lu dans le segment 0 ou 1, à
// l'adresse divisée par 2^up.
static void load_input(const int8_t* act_in, const LayerDesc& d, const TileInfo& t, int ti,
                       act_t in_buf[TN][IR][IC]) {
  const int rows = t.tr_n + d.k - 1, cols = t.tc_n + d.k - 1;
load_in:
  for (int n = 0; n < TN * IR * IC; ++n) {
#pragma HLS PIPELINE II=1
    const int tii = n / (IR * IC), rr = (n / IC) % IR, cc = n % IC;
    const int ch = ti + tii, ir = t.row - d.pad + rr, ic = t.col - d.pad + cc;
    act_t v = 0;
    if (ch < d.cin && rr < rows && cc < cols && ir >= 0 && ir < d.h && ic >= 0 && ic < d.w) {
      const bool s1 = ch >= d.seg0_c;
      const int c = s1 ? ch - d.seg0_c : ch;
      const int off = s1 ? d.seg1_off : d.seg0_off;
      const int sh = s1 ? d.seg1_h : d.seg0_h, sw = s1 ? d.seg1_w : d.seg0_w;
      const int up = s1 ? d.seg1_up : d.seg0_up;
      v = act_in[off + (c * sh + (ir >> up)) * sw + (ic >> up)];
    }
    in_buf[tii][rr][cc] = v;
  }
}

// load(w_buf) : poids (cout, cin, k, k) ; zéros au-delà de cout et de cin.
static void load_weights(const int8_t* wts, const LayerDesc& d, const TileInfo& t, int ti,
                         w_t w_buf[TM][TN][K_MAX][K_MAX]) {
  const int K = d.k, KK = K * K;
load_w:
  for (int n = 0; n < TM * TN * KK; ++n) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TM*TN max=TM*TN*K_MAX*K_MAX
    const int too = n / (TN * KK), tii = (n / KK) % TN, i = (n % KK) / K, j = n % K;
    const int o = t.to + too, c = ti + tii;
    w_t v = 0;
    if (o < d.cout && c < d.cin) v = wts[d.w_off + ((o * d.cin + c) * K + i) * K + j];
    w_buf[too][tii][i][j] = v;
  }
}

// Tm × Tn MAC par cycle : boucle (i, j, trr, tcc) aplatie en PIPELINE II=1, too et tii
// déroulés, arbre d'additions sur Tn. `first` : première ti, out_buf repart de zéro.
static void compute(const act_t in_buf[TN][IR][IC], const w_t w_buf[TM][TN][K_MAX][K_MAX],
                    acc_t out_buf[TM][TR][TC], int K, bool first) {
  int i = 0, j = 0, trr = 0, tcc = 0;
mac:
  for (int n = 0; n < K * K * TR * TC; ++n) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TR*TC max=K_MAX*K_MAX*TR*TC
#pragma HLS DEPENDENCE variable=out_buf inter false
    const bool init = first && i == 0 && j == 0;
    for (int too = 0; too < TM; ++too) {
#pragma HLS UNROLL
      acc_t sum = 0;
      for (int tii = 0; tii < TN; ++tii) {
#pragma HLS UNROLL
        sum += w_buf[too][tii][i][j] * in_buf[tii][trr + i][tcc + j];
      }
      out_buf[too][trr][tcc] = (init ? acc_t(0) : out_buf[too][trr][tcc]) + sum;
    }
    if (++tcc == TC) {
      tcc = 0;
      if (++trr == TR) {
        trr = 0;
        if (++j == K) {
          j = 0;
          ++i;
        }
      }
    }
  }
}

// Une tuile de sortie complète : accumulation sur ti avec double tampon in_buf / w_buf.
static void conv_tile(const int8_t* act_in, const int8_t* wts, const LayerDesc& d,
                      const TileInfo& t, act_t in_a[TN][IR][IC], act_t in_b[TN][IR][IC],
                      w_t w_a[TM][TN][K_MAX][K_MAX], w_t w_b[TM][TN][K_MAX][K_MAX],
                      acc_t out_buf[TM][TR][TC]) {
  const int nti = (d.cin + TN - 1) / TN;
  load_input(act_in, d, t, 0, in_a);
  load_weights(wts, d, t, 0, w_a);
ti_loop:
  for (int k = 0; k < nti; ++k) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=43
    const bool more = k + 1 < nti;
    const int ti_next = (k + 1) * TN;
    if (k % 2 == 0) {
      if (more) {
        load_input(act_in, d, t, ti_next, in_b);
        load_weights(wts, d, t, ti_next, w_b);
      }
      compute(in_a, w_a, out_buf, d.k, k == 0);
    } else {
      if (more) {
        load_input(act_in, d, t, ti_next, in_a);
        load_weights(wts, d, t, ti_next, w_a);
      }
      compute(in_b, w_b, out_buf, d.k, k == 0);
    }
  }
}

#ifndef __SYNTHESIS__
static void count_cycles(const LayerDesc& d, const TileInfo& t, bool first_tile,
                         uint64_t& prev_store) {
  const uint64_t KK = uint64_t(d.k) * d.k;
  const uint64_t nti = uint64_t(d.cin + TN - 1) / TN;
  const uint64_t ld_in = uint64_t(TN) * IR * IC, ld_w = uint64_t(TM) * TN * KK;
  const uint64_t ld = ld_in > ld_w ? ld_in : ld_w;  // deux bundles m_axi : en parallèle
  const uint64_t comp = KK * TR * TC;
  const int pk = d.pk(), ps = d.ps();
  const int Pr = (TR - pk) / ps + 1, Pc = (TC - pk) / ps + 1;
  const uint64_t own = d.pooled() && d.prepool_off >= 0
                           ? uint64_t(imin(t.tr_n, Pr * ps)) * imin(t.tc_n, Pc * ps)
                           : 0;
  const uint64_t st = uint64_t(t.tm_n) * (TR * TC + own + uint64_t(t.np_r) * t.np_c);

  SimCycles& s = sim_cycles;
  s.load_in += nti * ld_in;
  s.load_w += nti * ld_w;
  s.compute += nti * comp;
  s.store += st;
  s.sequential += nti * (ld_in + ld_w + comp) + st;
  // Ping-pong : chargement de ti = 0, puis max(chargement ti + 1, calcul ti).
  const uint64_t tile = ld + (nti - 1) * (ld > comp ? ld : comp) + comp;
  s.overlapped += first_tile ? tile : (tile > prev_store ? tile : prev_store);
  prev_store = st;
}
#endif

}  // namespace accel

using namespace accel;

void yolo_conv(const int8_t* act_in, int8_t* act_out, const int8_t* wts, const int32_t* prm,
               LayerDesc d) {
#pragma HLS INTERFACE m_axi port=act_in offset=slave bundle=gmem_in depth=1048576
#pragma HLS INTERFACE m_axi port=act_out offset=slave bundle=gmem_out depth=1048576
#pragma HLS INTERFACE m_axi port=wts offset=slave bundle=gmem_w depth=8845504
#pragma HLS INTERFACE m_axi port=prm offset=slave bundle=gmem_p depth=7392
#pragma HLS INTERFACE s_axilite port=act_in bundle=control
#pragma HLS INTERFACE s_axilite port=act_out bundle=control
#pragma HLS INTERFACE s_axilite port=wts bundle=control
#pragma HLS INTERFACE s_axilite port=prm bundle=control
#pragma HLS INTERFACE s_axilite port=d bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS AGGREGATE variable=d

  // Tampons sur puce en double (ping-pong) — 2 × (B_in + B_w + B_out), §10.2.
  static act_t in_a[TN][IR][IC], in_b[TN][IR][IC];
  static w_t w_a[TM][TN][K_MAX][K_MAX], w_b[TM][TN][K_MAX][K_MAX];
  static acc_t out_a[TM][TR][TC], out_b[TM][TR][TC];
#pragma HLS ARRAY_PARTITION variable=in_a complete dim=1
#pragma HLS ARRAY_PARTITION variable=in_b complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_a complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_a complete dim=2
#pragma HLS ARRAY_PARTITION variable=w_b complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_b complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_a complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_b complete dim=1

  const int R = d.out_h(), C = d.out_w();
  const int pk = d.pk(), ps = d.ps();
  const int Rp = d.pool_h(), Cp = d.pool_w();
  const int Pr = (TR - pk) / ps + 1, Pc = (TC - pk) / ps + 1;  // lignes poolées / tuile
  const int n_r = (Rp + Pr - 1) / Pr, n_c = (Cp + Pc - 1) / Pc, n_m = (d.cout + TM - 1) / TM;
  const int n_tiles = n_r * n_c * n_m;

#ifndef __SYNTHESIS__
  sim_cycles = SimCycles{};
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
