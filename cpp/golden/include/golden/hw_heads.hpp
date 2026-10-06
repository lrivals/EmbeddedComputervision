// Descripteurs de têtes du post-traitement matériel (T9.1) à partir du modèle exporté :
// ce que l'hôte écrit dans les registres de `yolo_post` (ancres Q8, seuil sur t_o, pas).
#pragma once

#include <stdexcept>

#include "golden/hw_postproc.hpp"
#include "golden/model.hpp"

namespace golden {

inline hwpp::HeadData make_hw_head(const Model& m, const Layer& l, const int8_t* data,
                                   double conf) {
  hwpp::HeadData hd;
  hwpp::HeadDesc& h = hd.desc;
  h.grid_h = l.out_h;
  h.grid_w = l.out_w;
  h.classes = m.classes;
  h.softmax = l.type == LayerType::Region;
  h.exp_frac = l.exp_frac;
  h.obj_thr_q = hwpp::logit_threshold_q(conf, l.scale);
  const int rw = hwpp::anchor_ref_w(m.in_h, m.in_w), rh = hwpp::anchor_ref_h(m.in_h, m.in_w);
  const int stride = rw / h.grid_w;
  if (stride * h.grid_w != rw || stride * h.grid_h != rh || (stride & (stride - 1)))
    throw std::runtime_error("post-traitement matériel : pas W/S_w = H/S_h non commun ou non "
                             "puissance de 2");
  while ((1 << h.stride_log2) < stride) ++h.stride_log2;
  std::vector<int> idx;
  if (h.softmax)
    for (int a = 0; a < l.num; ++a) idx.push_back(a);
  else
    idx = l.mask;
  if (int(idx.size()) > hwpp::MAX_ANCHORS || m.classes > hwpp::MAX_CLASSES)
    throw std::runtime_error("post-traitement matériel : trop d'ancres ou de classes");
  h.num_anchors = int(idx.size());
  for (int a = 0; a < h.num_anchors; ++a) {
    h.anchors[a][0] = hwpp::anchor_q8(m.anchors[size_t(idx[size_t(a)])].first);
    h.anchors[a][1] = hwpp::anchor_q8(m.anchors[size_t(idx[size_t(a)])].second);
  }
  hd.data = data;
  hd.sigmoid = m.head_lut(l, 0);
  hd.exp = m.head_lut(l, 1);
  hd.softmax_exp = m.head_lut(l, 2);
  return hd;
}

inline hwpp::Params make_hw_params(double conf, double iou) {
  hwpp::Params p;
  p.conf_q = hwpp::conf_q16(conf);
  hwpp::iou_fraction(iou, p.iou_p, p.iou_q);
  return p;
}

}  // namespace golden
