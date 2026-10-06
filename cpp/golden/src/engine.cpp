#include "golden/engine.hpp"

#include <stdexcept>

#include "golden/hw_postproc.hpp"

namespace golden {

ConvParams conv_params(const Layer& l, int in_h, int in_w) {
  if (l.s != S_CONV || l.k > K_MAX) throw std::runtime_error("conv hors du moteur (k, s)");
  ConvParams p;
  p.k = l.k;
  p.pad = l.pad;
  p.cin = l.cin;
  p.cout = l.cout;
  p.h = in_h;
  p.w = in_w;
  p.shift = l.shift;
  p.qmax = l.qmax;
  p.leaky = l.leaky;
  if (l.pool_layer >= 0) {
    p.pool_k = l.pool_k;
    p.pool_s = l.pool_s;
  }
  return p;
}

// Carte avant pooling d'une conv fusionnée (pooling 2×2 : s2 divise par 2, s1 conserve).
int prepool_h(const Layer& l) { return l.pool_s == 1 ? l.out_h : l.out_h * l.pool_s; }
int prepool_w(const Layer& l) { return l.pool_s == 1 ? l.out_w : l.out_w * l.pool_s; }

Engine::Engine(const Model& m) : m_(m) {
  for (const auto& kv : m.buffers) arena_[kv.first].assign(size_t(kv.second), 0);
  views_.resize(m.layers.size());
  stats_.resize(m.layers.size());
  taps_.resize(m.layers.size());
  for (const Layer& l : m.layers)
    if (l.type == LayerType::Maxpool && l.fused_into < 0)
      throw std::runtime_error("maxpool non fusionné : non pris en charge par le moteur");
  place_views();
}

void Engine::place_views() {
  for (const Layer& l : m_.layers) {
    InView& v = views_[size_t(l.id)];
    switch (l.type) {
      case LayerType::Conv: {
        const Layer* dst = l.pool_layer >= 0 ? &m_.layers[size_t(l.pool_layer)] : &l;
        v = single_view(addr(l.out), dst->out_c, dst->out_h, dst->out_w);
        if (l.pool_layer >= 0) {
          // La conv elle-même : sa carte avant pooling si elle est en DDR (`prepool_out`).
          views_[size_t(l.pool_layer)] = v;
          v = l.prepool_out.valid()
                  ? single_view(addr(l.prepool_out), l.cout, prepool_h(l), prepool_w(l))
                  : InView{};
        }
        break;
      }
      case LayerType::Maxpool:
        break;  // posée par sa conv
      case LayerType::Upsample: {
        v = views_[size_t(l.id - 1)];
        if (v.nseg != 1) throw std::runtime_error("upsample d'une vue composée");
        v.seg[0].up += 1;  // s = 2 : division d'adresse par 2
        if (l.s != 2) throw std::runtime_error("upsample ×2 seulement");
        v.h *= 2;
        v.w *= 2;
        break;
      }
      case LayerType::Route: {
        v = InView{};
        for (int j : l.from) {
          const InView& s = views_[size_t(j)];
          if (s.nseg == 0) throw std::runtime_error("source de route sans sortie en DDR");
          if (v.nseg > 0 && (s.h != v.h || s.w != v.w))
            throw std::runtime_error("route : formes incompatibles");
          v.h = s.h;
          v.w = s.w;
          for (int k = 0; k < s.nseg; ++k) {
            Segment g = s.seg[k];
            Segment* last = v.nseg ? &v.seg[v.nseg - 1] : nullptr;
            // Placement contigu : fusion avec le segment précédent (concaténation gratuite).
            if (last && last->up == 0 && g.up == 0 && last->h == g.h && last->w == g.w &&
                last->base + int64_t(last->c) * last->h * last->w == g.base) {
              last->c += g.c;
            } else {
              if (v.nseg == MAX_SEGMENTS) throw std::runtime_error("route : trop de segments");
              v.seg[v.nseg++] = g;
            }
            v.c += g.c;
          }
        }
        break;
      }
      case LayerType::Yolo:
      case LayerType::Region:
        v = views_[size_t(l.id - 1)];
        break;
    }
    if (v.nseg > 0 && (v.c != l.out_c || v.h != l.out_h || v.w != l.out_w) &&
        !(l.type == LayerType::Conv && l.pool_layer >= 0))
      throw std::runtime_error("vue de la couche " + std::to_string(l.id) +
                               " : forme différente de out_shape");
  }
}

InView Engine::in_view(const Layer& l) const {
  InView v = l.id == 0 ? single_view(arena_.at(m_.input.buf).data() + m_.input.offset, m_.in_c,
                                     m_.in_h, m_.in_w)
                       : views_[size_t(l.id - 1)];
  if (v.c != l.cin) throw std::runtime_error("conv " + std::to_string(l.id) + " : cin");
  // Entrée simple : elle doit être là où le manifest l'attend.
  if (v.nseg == 1 && v.seg[0].up == 0 &&
      v.seg[0].base != arena_.at(l.in.buf).data() + l.in.offset)
    throw std::runtime_error("conv " + std::to_string(l.id) + " : entrée hors de `in`");
  return v;
}

Tensor Engine::materialize(const InView& v) const {
  Tensor t;
  t.c = v.c;
  t.h = v.h;
  t.w = v.w;
  t.data.resize(size_t(v.c) * v.h * v.w);
  size_t a = 0;
  for (int c = 0; c < v.c; ++c)
    for (int r = 0; r < v.h; ++r)
      for (int q = 0; q < v.w; ++q) t.data[a++] = v.at(c, r, q);
  return t;
}

void Engine::record(int id) {
  const Layer& l = m_.layers[size_t(id)];
  if (l.type == LayerType::Conv && l.pool_layer >= 0) {
    // Carte avant pooling : la sonde de débogage.
    rec_[size_t(id)] = Tensor{l.cout, prepool_h(l), prepool_w(l), taps_[size_t(id)]};
    return;
  }
  rec_[size_t(id)] = materialize(views_[size_t(id)]);
}

std::vector<postproc::Head> make_heads(const Engine& e) {
  const Model& m = e.model();
  std::vector<postproc::Head> heads;
  for (int id : m.heads()) {
    const Layer& l = m.layers[size_t(id)];
    const InView& v = e.view(id);
    if (v.nseg != 1 || v.seg[0].up != 0) throw std::runtime_error("tête non contiguë");
    postproc::Head h;
    h.data = v.seg[0].base;
    h.grid_h = l.out_h;
    h.grid_w = l.out_w;
    h.ref_w = hwpp::anchor_ref_w(m.in_h, m.in_w);
    h.ref_h = hwpp::anchor_ref_h(m.in_h, m.in_w);
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

}  // namespace golden
