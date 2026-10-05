// Table de paramètres du noyau `yolo_post` (T9.1) : tables σ, e^t, e^{d·s} puis un
// descripteur par tête (hls/kernels/postproc.hpp). Partagé par le testbench et le driver.
#pragma once

#include <vector>

#include "golden/hw_heads.hpp"
#include "golden/model.hpp"
#include "postproc.hpp"

namespace driver {

struct PostTable {
  std::vector<int32_t> words;  // à placer à l'indice `base` du tableau `tab`
  accel::PostDesc desc{};
};

// `head_off[k]` : octets de la k-ième tête (ordre `Model::heads`) dans l'arène ;
// `base` : indice du premier mot de la table dans `tab` ; `out_off` : indice dans `res`.
inline PostTable post_table(const golden::Model& m, const std::vector<int64_t>& head_off,
                            double conf, double iou, int32_t base = 0, int32_t out_off = 0) {
  const std::vector<int> ids = m.heads();
  if (ids.size() != head_off.size() || int(ids.size()) > accel::POST_MAX_HEADS)
    throw std::runtime_error("post_table : têtes");
  PostTable t;
  const int nh = int(ids.size());
  t.words.assign(size_t(nh) * (3 * 256 + accel::POST_HEAD_WORDS), 0);
  for (int k = 0; k < nh; ++k) {
    const golden::Layer& l = m.layers[size_t(ids[size_t(k)])];
    const hwpp::HeadData hd = golden::make_hw_head(m, l, nullptr, conf);
    const size_t lut = size_t(k) * 3 * 256;
    for (int w = 0; w < 3; ++w)
      for (int i = 0; i < 256; ++i)
        t.words[lut + size_t(w) * 256 + size_t(i)] = int32_t(m.head_lut(l, w)[i]);
    int32_t* d = t.words.data() + size_t(nh) * 3 * 256 + size_t(k) * accel::POST_HEAD_WORDS;
    d[accel::PD_DATA_OFF] = int32_t(head_off[size_t(k)]);
    d[accel::PD_GRID] = hd.desc.grid;
    d[accel::PD_CLASSES] = hd.desc.classes;
    d[accel::PD_ANCHORS] = hd.desc.num_anchors;
    d[accel::PD_STRIDE_LOG2] = hd.desc.stride_log2;
    d[accel::PD_EXP_FRAC] = hd.desc.exp_frac;
    d[accel::PD_OBJ_THR] = hd.desc.obj_thr_q;
    d[accel::PD_SOFTMAX] = hd.desc.softmax;
    d[accel::PD_LUT_OFF] = base + int32_t(lut);
    for (int a = 0; a < hwpp::MAX_ANCHORS; ++a) {
      d[accel::PD_ANCHOR0 + 2 * a] = hd.desc.anchors[a][0];
      d[accel::PD_ANCHOR0 + 2 * a + 1] = hd.desc.anchors[a][1];
    }
  }
  const hwpp::Params p = golden::make_hw_params(conf, iou);
  t.desc = accel::PostDesc{nh, base + nh * 3 * 256, p.conf_q, p.iou_p, p.iou_q, out_off};
  return t;
}

// Résultat de `yolo_post` → boîtes (et débordements).
inline std::vector<hwpp::Box> post_boxes(const int32_t* res, int* overflow = nullptr) {
  std::vector<hwpp::Box> out;
  for (int k = 0; k < res[0]; ++k) {
    const int32_t* o = res + 2 + accel::POST_BOX_WORDS * k;
    out.push_back(hwpp::Box{o[0], o[1], o[2], o[3], o[4], o[5]});
  }
  if (overflow) *overflow = res[1];
  return out;
}

}  // namespace driver
