// Post-traitement matériel (T9.1.2) : décodage par tables et NMS sans tri [2024-zhang#014.2].
//
// Flux : pour chaque tête, chaque ancre, chaque cellule (II = 1) on lit t_o ; seules les
// survivantes (t_o ≥ seuil) lisent les 4 + C autres canaux, passent par les tables et
// produisent des candidates (cellule, classe), poussées une à une dans le sélecteur.
// L'arithmétique est celle de golden/hw_postproc.hpp (decode_cell, cell_scores, Selector).
#include "postproc.hpp"

namespace accel {

#ifndef __SYNTHESIS__
PostCycles post_cycles;
#endif

}  // namespace accel

using namespace accel;

void yolo_post(const int8_t* act, const int32_t* tab, int32_t* res, PostDesc d) {
#pragma HLS INTERFACE m_axi port=act offset=slave bundle=gmem_in depth=1048576
#pragma HLS INTERFACE m_axi port=tab offset=slave bundle=gmem_p depth=8192
#pragma HLS INTERFACE m_axi port=res offset=slave bundle=gmem_out depth=1538
#pragma HLS INTERFACE s_axilite port=act bundle=control
#pragma HLS INTERFACE s_axilite port=tab bundle=control
#pragma HLS INTERFACE s_axilite port=res bundle=control
#pragma HLS INTERFACE s_axilite port=d bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS AGGREGATE variable=d

  static uint32_t sig[256], ex[256], smexp[256];
  static hwpp::Selector<POST_CAP> sel;
#pragma HLS ARRAY_PARTITION variable=sel.sel cyclic factor=4
  sel.n = 0;
  sel.overflow = 0;

#ifndef __SYNTHESIS__
  post_cycles = PostCycles{};
  PostCycles& cyc = post_cycles;
#endif

heads:
  for (int hn = 0; hn < d.nheads && hn < POST_MAX_HEADS; ++hn) {
    const int32_t* hd = tab + d.desc_off + hn * POST_HEAD_WORDS;
    hwpp::HeadDesc h;
    h.grid = hd[PD_GRID];
    h.classes = hd[PD_CLASSES];
    h.num_anchors = hd[PD_ANCHORS];
    h.stride_log2 = hd[PD_STRIDE_LOG2];
    h.exp_frac = hd[PD_EXP_FRAC];
    h.obj_thr_q = hd[PD_OBJ_THR];
    h.softmax = hd[PD_SOFTMAX] != 0;
    for (int a = 0; a < hwpp::MAX_ANCHORS; ++a) {
      h.anchors[a][0] = hd[PD_ANCHOR0 + 2 * a];
      h.anchors[a][1] = hd[PD_ANCHOR0 + 2 * a + 1];
    }
    const int lut = hd[PD_LUT_OFF];
  load_luts:
    for (int k = 0; k < 256; ++k) {
#pragma HLS PIPELINE II=1
      sig[k] = uint32_t(tab[lut + k]);
      ex[k] = uint32_t(tab[lut + 256 + k]);
      smexp[k] = uint32_t(tab[lut + 512 + k]);
    }
#ifndef __SYNTHESIS__
    cyc.luts += 256;
#endif
    const int s = h.grid, nc = h.classes, plane = s * s;
    const int8_t* base = act + hd[PD_DATA_OFF];
  anchors:
    for (int a = 0; a < h.num_anchors; ++a) {
      const int8_t* p = base + a * (5 + nc) * plane;
    cells:
      for (int cell = 0; cell < plane; ++cell) {
#pragma HLS PIPELINE II=1
#ifndef __SYNTHESIS__
        ++cyc.cells;
        ++cyc.scan;
#endif
        if (int(p[4 * plane + cell]) < h.obj_thr_q) continue;
        int t[hwpp::MAX_CLASSES];
        int32_t score[hwpp::MAX_CLASSES];
        for (int k = 0; k < nc; ++k) t[k] = p[(5 + k) * plane + cell];
        hwpp::cell_scores(h, int(p[4 * plane + cell]), t, sig, smexp, score);
        hwpp::Box b;
        hwpp::decode_cell(h, a, cell / s, cell % s, p[cell], p[plane + cell],
                          p[2 * plane + cell], p[3 * plane + cell], sig, ex, b);
#ifndef __SYNTHESIS__
        ++cyc.survivors;
        // lectures 4 + C, puis max, somme (softmax) et scores : une classe par cycle
        cyc.decode += uint64_t(4 + nc) + (h.softmax ? 2 * nc : 0) + nc;
#endif
        for (int k = 0; k < nc; ++k)
          if (score[k] > d.conf_q) {
            b.score = score[k];
            b.cls = k;
#ifndef __SYNTHESIS__
            ++cyc.candidates;
            cyc.nms += uint64_t(sel.n > 0 ? sel.n : 1);
#endif
            sel.push(b, d.iou_p, d.iou_q);
          }
      }
    }
  }

  int n = 0;
store:
  for (int k = 0; k < sel.n; ++k) {
#pragma HLS PIPELINE II=6
    if (!sel.alive[k]) continue;
    int32_t* o = res + d.out_off + 2 + POST_BOX_WORDS * n;
    const hwpp::Box& b = sel.sel[k];
    o[0] = b.x1;
    o[1] = b.y1;
    o[2] = b.x2;
    o[3] = b.y2;
    o[4] = b.score;
    o[5] = b.cls;
    ++n;
  }
  res[d.out_off] = n;
  res[d.out_off + 1] = sel.overflow;
#ifndef __SYNTHESIS__
  cyc.store = uint64_t(POST_BOX_WORDS) * n + 2;
  cyc.total = cyc.luts + cyc.scan + cyc.decode + cyc.nms + cyc.store;
#endif
}
