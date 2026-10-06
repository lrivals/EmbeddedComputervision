// Post-traitement des têtes int8 sur l'hôte : seuil sur le logit, LUT, décodage, NMS
// (§8.1, §8.2, §9.4).
//
// En-tête seul qui ne dépend que de la bibliothèque standard : réutilisable tel quel dans
// sw/postproc (ARM). Réplique opération par opération `decode_head_int` et `filter_and_nms`
// (python/yolo/infer/{decode,nms}.py) : les doubles sont égaux bit à bit à ceux de Python.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace postproc {

constexpr double ANCHOR_REF = 416.0;  // ancres en pixels pour une entrée de 416 (carrée)
constexpr double LUT_ONE = 65536.0;   // tables σ et softmax en Q16
constexpr int LUT_OFFSET = 128;       // index = q + 128
constexpr int SOFTMAX_OFFSET = 255;   // index = d + 255, d = q_c − q_max ∈ [−255, 0]

// Une tête int8 (A·(5+C), S_h, S_w) d'échelle `scale` et ses tables (luts.bin).
struct Head {
  const int8_t* data = nullptr;
  int grid_h = 0, grid_w = 0;
  // (W, H) de référence des ancres : 416 × 416 (entrée carrée), width × height sinon (T11.4).
  double ref_w = ANCHOR_REF, ref_h = ANCHOR_REF;
  int classes = 0;
  bool softmax = false;  // region (YOLOv2) : softmax ; yolo (YOLOv3) : sigmoïdes
  double scale = 0.125;
  int exp_frac = 16;
  const uint32_t* sigmoid = nullptr;      // σ(q s) en Q16
  const uint32_t* exp = nullptr;          // e^{q s} en Q`exp_frac`
  const uint32_t* softmax_exp = nullptr;  // e^{d s} en Q16
  std::vector<std::pair<double, double>> anchors;  // pixels (repère ref_w × ref_h)
};

// Boîte (cx, cy, w, h) normalisée dans l'image d'entrée.
struct Candidate {
  double box[4];
  std::vector<double> scores;  // σ(t_o) · p_c
};

struct Detection {
  double box[4];
  double score;
  int label;
};

// §9.4 : plus petit entier q tel que σ(q s) > θ, soit ⌊logit(θ)/s⌋ + 1.
inline int logit_threshold_q(double theta, double scale) {
  return int(std::floor(std::log(theta / (1.0 - theta)) / scale)) + 1;
}

// §8.1, §9.4 : cellules dont l'entier t_o passe le seuil, dans l'ordre (ancre, ligne,
// colonne) ; les tables ne servent qu'aux survivantes.
inline void decode_head(const Head& h, double obj_thr, std::vector<Candidate>& out) {
  const int gh = h.grid_h, gw = h.grid_w, nc = h.classes, a_n = int(h.anchors.size());
  const int thr = logit_threshold_q(obj_thr, h.scale);
  const size_t plane = size_t(gh) * gw;
  for (int a = 0; a < a_n; ++a) {
    const int8_t* p = h.data + size_t(a) * (5 + nc) * plane;
    const double aw = h.anchors[size_t(a)].first / h.ref_w;
    const double ah = h.anchors[size_t(a)].second / h.ref_h;
    for (int i = 0; i < gh; ++i)
      for (int j = 0; j < gw; ++j) {
        const size_t cell = size_t(i) * gw + j;
        auto t = [&](int ch) { return int(p[size_t(ch) * plane + cell]); };
        if (t(4) < thr) continue;
        Candidate c;
        // §8.1 : b_x = (σ(t_x) + j)/S_w, b_w = p_w e^{t_w}
        c.box[0] = (h.sigmoid[t(0) + LUT_OFFSET] / LUT_ONE + j) / gw;
        c.box[1] = (h.sigmoid[t(1) + LUT_OFFSET] / LUT_ONE + i) / gh;
        const double ef = std::ldexp(1.0, h.exp_frac);
        c.box[2] = aw * (h.exp[t(2) + LUT_OFFSET] / ef);
        c.box[3] = ah * (h.exp[t(3) + LUT_OFFSET] / ef);
        const double obj = h.sigmoid[t(4) + LUT_OFFSET] / LUT_ONE;
        c.scores.resize(size_t(nc));
        if (h.softmax) {
          // Softmax en trois étages : max entier, e^{(q − q_max) s} par table, division.
          int qmax = t(5);
          for (int k = 1; k < nc; ++k) qmax = std::max(qmax, t(5 + k));
          int64_t sum = 0;
          for (int k = 0; k < nc; ++k) sum += h.softmax_exp[t(5 + k) - qmax + SOFTMAX_OFFSET];
          for (int k = 0; k < nc; ++k)
            c.scores[size_t(k)] =
                obj * (h.softmax_exp[t(5 + k) - qmax + SOFTMAX_OFFSET] / double(sum));
        } else {
          for (int k = 0; k < nc; ++k)
            c.scores[size_t(k)] = obj * (h.sigmoid[t(5 + k) + LUT_OFFSET] / LUT_ONE);
        }
        out.push_back(std::move(c));
      }
  }
}

// §5.3 : IoU de deux boîtes (cx, cy, w, h), comme `yolo.infer.boxes.iou`.
inline double iou(const double* a, const double* b) {
  const double ax1 = a[0] - a[2] / 2, ay1 = a[1] - a[3] / 2;
  const double ax2 = a[0] + a[2] / 2, ay2 = a[1] + a[3] / 2;
  const double bx1 = b[0] - b[2] / 2, by1 = b[1] - b[3] / 2;
  const double bx2 = b[0] + b[2] / 2, by2 = b[1] + b[3] / 2;
  const double iw = std::max(0.0, std::min(ax2, bx2) - std::max(ax1, bx1));
  const double ih = std::max(0.0, std::min(ay2, by2) - std::max(ay1, by1));
  const double inter = iw * ih;
  const double uni = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
  return uni > 0 ? inter / uni : 0.0;
}

// §8.2 : indices gardés par score décroissant (tri stable) ; supprimée si IoU > seuil.
inline std::vector<size_t> nms(const std::vector<const double*>& boxes,
                               const std::vector<double>& scores, double iou_thr) {
  std::vector<size_t> order(scores.size());
  for (size_t k = 0; k < order.size(); ++k) order[k] = k;
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t x, size_t y) { return scores[x] > scores[y]; });
  std::vector<bool> alive(order.size(), true);
  std::vector<size_t> keep;
  for (size_t k = 0; k < order.size(); ++k) {
    if (!alive[k]) continue;
    keep.push_back(order[k]);
    for (size_t q = k + 1; q < order.size(); ++q)
      if (alive[q] && !(iou(boxes[order[k]], boxes[order[q]]) <= iou_thr)) alive[q] = false;
  }
  return keep;
}

// §8.2 : seuil puis NMS classe par classe ; résultat trié par score décroissant (stable).
inline std::vector<Detection> filter_and_nms(const std::vector<Candidate>& cand, int classes,
                                             double conf_thr, double iou_thr) {
  std::vector<Detection> dets;
  for (int c = 0; c < classes; ++c) {
    std::vector<const double*> boxes;
    std::vector<double> scores;
    std::vector<size_t> idx;
    for (size_t k = 0; k < cand.size(); ++k)
      if (cand[k].scores[size_t(c)] > conf_thr) {
        boxes.push_back(cand[k].box);
        scores.push_back(cand[k].scores[size_t(c)]);
        idx.push_back(k);
      }
    for (size_t k : nms(boxes, scores, iou_thr)) {
      const Candidate& src = cand[idx[k]];
      dets.push_back(Detection{{src.box[0], src.box[1], src.box[2], src.box[3]}, scores[k], c});
    }
  }
  std::stable_sort(dets.begin(), dets.end(),
                   [](const Detection& x, const Detection& y) { return x.score > y.score; });
  return dets;
}

// Têtes d'une image → détections. Le seuil sur t_o reprend `conf_thr` (score ≤ objectness).
inline std::vector<Detection> postprocess(const std::vector<Head>& heads, double conf_thr = 0.25,
                                          double iou_thr = 0.45) {
  std::vector<Candidate> cand;
  int classes = 0;
  for (const Head& h : heads) {
    decode_head(h, conf_thr, cand);
    classes = h.classes;
  }
  return filter_and_nms(cand, classes, conf_thr, iou_thr);
}

}  // namespace postproc
