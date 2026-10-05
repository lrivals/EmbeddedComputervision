#include "heads.hpp"

#include <algorithm>
#include <stdexcept>

namespace sw {

using golden::Layer;
using golden::LayerType;

std::vector<postproc::Head> make_heads(const golden::Model& m, const driver::Program& p,
                                       const int8_t* arena) {
  std::vector<postproc::Head> heads;
  for (int id : m.heads()) {
    const Layer& l = m.layers[size_t(id)];
    const driver::View& v = p.views[size_t(id)];
    if (v.nseg != 1 || v.seg[0].up != 0) throw std::runtime_error("tête non contiguë");
    postproc::Head h;
    h.data = arena + v.seg[0].off;
    h.grid = l.out_h;
    h.classes = m.classes;
    h.softmax = l.type == LayerType::Region;
    h.scale = l.scale;
    h.exp_frac = l.exp_frac;
    h.sigmoid = m.head_lut(l, 0);
    h.exp = m.head_lut(l, 1);
    h.softmax_exp = m.head_lut(l, 2);
    if (h.softmax)
      for (int a = 0; a < l.num; ++a) h.anchors.push_back(m.anchors[size_t(a)]);
    else
      for (int a : l.mask) h.anchors.push_back(m.anchors[size_t(a)]);
    if (int(h.anchors.size()) * (5 + m.classes) != l.out_c)
      throw std::runtime_error("tête : canaux != A·(5+C)");
    heads.push_back(std::move(h));
  }
  return heads;
}

std::vector<postproc::Detection> hw_detections(const std::vector<hwpp::Box>& boxes) {
  const double unit = double(hwpp::ANCHOR_REF << hwpp::BOX_FRAC);
  std::vector<postproc::Detection> dets;
  for (const hwpp::Box& b : boxes)
    dets.push_back(postproc::Detection{{(double(b.x1) + b.x2) / 2 / unit,
                                        (double(b.y1) + b.y2) / 2 / unit,
                                        double(b.x2 - b.x1) / unit, double(b.y2 - b.y1) / unit},
                                       b.score / double(1 << hwpp::LUT_FRAC), b.cls});
  std::stable_sort(dets.begin(), dets.end(), [](const auto& x, const auto& y) {
    return x.score > y.score;
  });
  return dets;
}

}  // namespace sw
