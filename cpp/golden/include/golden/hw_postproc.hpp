// Post-traitement tout entier, tel que le matériel le fait (T9.1.1, §8.1, §8.2, §9.4, §10.3).
//
// Miroir opération par opération de python/yolo/infer/hw_postproc.py (convention détaillée
// dans sa docstring) : seuil sur l'entier t_o, tables Q16, coins en pixels Q4, scores Q16
// (une réciproque par cellule pour le softmax v2), IoU par produits croisés et NMS sans tri
// de 2024-zhang [2024-zhang#014.2].
//
// Pas de flottant ni d'allocation dans les primitives (`decode_cell`, `overlaps`,
// `Selector`) : le noyau HLS hls/kernels/postproc.cpp les reprend telles quelles. Seule
// `run` (référence) utilise la bibliothèque standard.
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace hwpp {

constexpr int ANCHOR_REF = 416;  // ancres en pixels pour une entrée de 416
constexpr int ANCHOR_FRAC = 8;   // ancres en Q8
constexpr int BOX_FRAC = 4;      // coins en pixels Q4
constexpr int LUT_FRAC = 16;     // tables σ et softmax en Q16
constexpr int LUT_OFFSET = 128;
constexpr int SOFTMAX_OFFSET = 255;
constexpr int64_t W_MAX = (int64_t(1) << 20) - 1;
constexpr int MAX_ANCHORS = 8;
constexpr int MAX_CLASSES = 80;
// Emplacements de sélection du noyau ; -DHWPP_CAP=N pour chiffrer une autre capacité en
// C-sim (T12.7, tb_post compilé à part).
#ifndef HWPP_CAP
#define HWPP_CAP 256
#endif
constexpr int CAP = HWPP_CAP;

// Boîte retenue : coins en pixels Q4, score Q16, classe.
struct Box {
  int32_t x1, y1, x2, y2;
  int32_t score;
  int32_t cls;
};

// Ce que le noyau reçoit pour une tête (registres) ; les tables sont à part.
struct HeadDesc {
  int grid = 0, classes = 0, num_anchors = 0;
  int stride_log2 = 0;  // log2(416 / grid)
  int exp_frac = 0;
  int obj_thr_q = 0;    // plus petit t_o gardé (§9.4)
  bool softmax = false;
  int32_t anchors[MAX_ANCHORS][2] = {};  // Q8
};

struct Params {
  int32_t conf_q = 0;  // score gardé si > conf_q (Q16)
  int iou_p = 9, iou_q = 20;  // θ = p/q
};

inline int64_t rshift_round(int64_t v, int n) {
  return n > 0 ? (v + (int64_t(1) << (n - 1))) >> n : v;
}

// ---------------------------------------------------------------- réglages côté hôte
inline int32_t conf_q16(double conf) { return int32_t(std::floor(conf * (1 << LUT_FRAC))); }

inline int32_t anchor_q8(double px) { return int32_t(std::floor(px * (1 << ANCHOR_FRAC) + 0.5)); }

// θ = p/q de plus petit dénominateur ≤ 1000 (0,45 → 9/20), comme `iou_fraction` en Python.
inline void iou_fraction(double iou, int& p, int& q) {
  for (q = 1; q <= 1000; ++q) {
    p = int(std::floor(iou * q + 0.5));
    if (std::fabs(double(p) / q - iou) < 1e-12) return;
  }
  q = 1000;
  p = int(std::floor(iou * q + 0.5));
}

inline int logit_threshold_q(double theta, double scale) {
  return int(std::floor(std::log(theta / (1.0 - theta)) / scale)) + 1;
}

// ------------------------------------------------------------------------ primitives
// Coins (Q4) de la cellule (a, i, j) à partir de t_x, t_y, t_w, t_h.
inline void decode_cell(const HeadDesc& h, int a, int i, int j, int tx, int ty, int tw, int th,
                        const uint32_t* sig, const uint32_t* ex, Box& b) {
  const int sh = 16 - BOX_FRAC, wsh = ANCHOR_FRAC + h.exp_frac - BOX_FRAC;
  const int64_t cx = rshift_round((int64_t(sig[tx + LUT_OFFSET]) + (int64_t(j) << 16))
                                  << h.stride_log2, sh);
  const int64_t cy = rshift_round((int64_t(sig[ty + LUT_OFFSET]) + (int64_t(i) << 16))
                                  << h.stride_log2, sh);
  int64_t w = rshift_round(int64_t(h.anchors[a][0]) * ex[tw + LUT_OFFSET], wsh);
  int64_t hh = rshift_round(int64_t(h.anchors[a][1]) * ex[th + LUT_OFFSET], wsh);
  if (w > W_MAX) w = W_MAX;
  if (hh > W_MAX) hh = W_MAX;
  b.x1 = int32_t(cx - (w >> 1));
  b.y1 = int32_t(cy - (hh >> 1));
  b.x2 = int32_t(b.x1 + w);
  b.y2 = int32_t(b.y1 + hh);
}

// Scores Q16 des classes d'une cellule ; `t` : t_c des C classes.
inline void cell_scores(const HeadDesc& h, int to, const int* t, const uint32_t* sig,
                        const uint32_t* smexp, int32_t* score) {
  const int64_t obj = sig[to + LUT_OFFSET];
  if (h.softmax) {
    // Softmax en trois étages : max entier, tables, une réciproque par cellule.
    int qmax = t[0];
    for (int k = 1; k < h.classes; ++k) qmax = t[k] > qmax ? t[k] : qmax;
    int64_t sum = 0;
    for (int k = 0; k < h.classes; ++k) sum += smexp[t[k] - qmax + SOFTMAX_OFFSET];
    const int64_t recip = (int64_t(1) << 32) / sum;
    for (int k = 0; k < h.classes; ++k)
      score[k] = int32_t(rshift_round(obj * smexp[t[k] - qmax + SOFTMAX_OFFSET] * recip, 32));
  } else {
    for (int k = 0; k < h.classes; ++k)
      score[k] = int32_t(rshift_round(obj * sig[t[k] + LUT_OFFSET], LUT_FRAC));
  }
}

// IoU(a, b) > p/q ⟺ (p + q)·inter > p·(aire_a + aire_b).
inline bool overlaps(const Box& a, const Box& b, int p, int q) {
  const int64_t iw = int64_t(a.x2 < b.x2 ? a.x2 : b.x2) - (a.x1 > b.x1 ? a.x1 : b.x1);
  const int64_t ih = int64_t(a.y2 < b.y2 ? a.y2 : b.y2) - (a.y1 > b.y1 ? a.y1 : b.y1);
  const int64_t inter = (iw > 0 ? iw : 0) * (ih > 0 ? ih : 0);
  const int64_t area_a = int64_t(a.x2 - a.x1) * (a.y2 - a.y1);
  const int64_t area_b = int64_t(b.x2 - b.x1) * (b.y2 - b.y1);
  return (p + q) * inter > p * (area_a + area_b);
}

// NMS sans tri : emplacements de sélection, remplis dans l'ordre du flux.
template <int N>
struct Selector {
  Box sel[N];
  bool alive[N];
  int n = 0;
  int overflow = 0;

  void push(const Box& b, int p, int q) {
    int first = -1;
    bool beaten = false;
    for (int k = 0; k < n; ++k)
      if (alive[k] && sel[k].cls == b.cls && overlaps(b, sel[k], p, q)) {
        if (sel[k].score >= b.score) beaten = true;
        if (first < 0) first = k;
      }
    if (beaten) return;  // recouverte par une meilleure (ou égale, arrivée avant)
    if (first >= 0) {
      for (int k = first + 1; k < n; ++k)
        if (alive[k] && sel[k].cls == b.cls && overlaps(b, sel[k], p, q)) alive[k] = false;
      sel[first] = b;
    } else if (n < N) {
      sel[n] = b;
      alive[n++] = true;
    } else {
      ++overflow;
    }
  }
};

// --------------------------------------------------------------------------- référence
struct HeadData {
  HeadDesc desc;
  const int8_t* data = nullptr;  // (A·(5+C), S, S)
  const uint32_t* sigmoid = nullptr;
  const uint32_t* exp = nullptr;
  const uint32_t* softmax_exp = nullptr;
};

// Candidates d'une tête, dans l'ordre (ancre, ligne, colonne, classe).
template <class F>
inline void for_each_candidate(const HeadData& hd, const Params& prm, F&& emit) {
  const HeadDesc& h = hd.desc;
  const int s = h.grid, nc = h.classes;
  const size_t plane = size_t(s) * s;
  int t[MAX_CLASSES];
  int32_t score[MAX_CLASSES];
  for (int a = 0; a < h.num_anchors; ++a) {
    const int8_t* p = hd.data + size_t(a) * (5 + nc) * plane;
    for (int i = 0; i < s; ++i)
      for (int j = 0; j < s; ++j) {
        const size_t cell = size_t(i) * s + j;
        auto q = [&](int ch) { return int(p[size_t(ch) * plane + cell]); };
        if (q(4) < h.obj_thr_q) continue;
        for (int k = 0; k < nc; ++k) t[k] = q(5 + k);
        cell_scores(h, q(4), t, hd.sigmoid, hd.softmax_exp, score);
        Box b{};
        decode_cell(h, a, i, j, q(0), q(1), q(2), q(3), hd.sigmoid, hd.exp, b);
        for (int k = 0; k < nc; ++k)
          if (score[k] > prm.conf_q) {
            b.score = score[k];
            b.cls = k;
            emit(b);
          }
      }
  }
}

// Têtes d'une image → boîtes retenues dans l'ordre des emplacements.
template <int N = CAP>
inline std::vector<Box> run(const std::vector<HeadData>& heads, const Params& prm,
                            int* overflow = nullptr) {
  auto* s = new Selector<N>();
  for (const HeadData& hd : heads)
    for_each_candidate(hd, prm, [&](const Box& b) { s->push(b, prm.iou_p, prm.iou_q); });
  std::vector<Box> out;
  for (int k = 0; k < s->n; ++k)
    if (s->alive[k]) out.push_back(s->sel[k]);
  if (overflow) *overflow = s->overflow;
  delete s;
  return out;
}

}  // namespace hwpp
