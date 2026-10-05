#include "program.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "golden/engine.hpp"
#include "weight_layout.hpp"

namespace driver {

using golden::Layer;
using golden::LayerType;

View single_view(int64_t off, int c, int h, int w) {
  View v;
  v.seg[0] = Segment{off, c, h, w, 0};
  v.nseg = 1;
  v.c = c;
  v.h = h;
  v.w = w;
  return v;
}

int8_t view_at(const int8_t* arena, const View& v, int ch, int r, int col) {
  for (int s = 0; s < v.nseg; ++s) {
    const Segment& g = v.seg[s];
    if (ch < g.c) return arena[g.off + (int64_t(ch) * g.h + (r >> g.up)) * g.w + (col >> g.up)];
    ch -= g.c;
  }
  throw std::out_of_range("view_at : canal hors de la vue");
}

std::vector<int8_t> materialize(const int8_t* arena, const View& v) {
  std::vector<int8_t> t(size_t(v.c) * v.h * v.w);
  size_t a = 0;
  for (int c = 0; c < v.c; ++c)
    for (int r = 0; r < v.h; ++r)
      for (int q = 0; q < v.w; ++q) t[a++] = view_at(arena, v, c, r, q);
  return t;
}

static int32_t i32(int64_t x, const char* what) {
  if (x < std::numeric_limits<int32_t>::min() || x > std::numeric_limits<int32_t>::max())
    throw std::runtime_error(std::string("registre 32 bits dépassé : ") + what);
  return int32_t(x);
}

accel::LayerDesc conv_desc(const Layer& l, const View& in, int64_t out_off, int64_t prepool_off,
                           int64_t b_index, int64_t m0_index, int64_t w_off) {
  const golden::ConvParams p = golden::conv_params(l, in.h, in.w);
  if (p.pooled() && (p.pool_k > 2 || p.pool_s > 2))
    throw std::runtime_error("maxpool fusionné : 2×2 seulement");
  if (in.nseg < 1 || in.nseg > accel::MAX_SEGMENTS || in.c != l.cin)
    throw std::runtime_error("conv " + std::to_string(l.id) + " : entrée invalide");
  accel::LayerDesc d{};
  d.k = p.k;
  d.pad = p.pad;
  d.cin = p.cin;
  d.cout = p.cout;
  d.h = p.h;
  d.w = p.w;
  d.shift = p.shift;
  d.qmax = p.qmax;
  d.leaky = p.leaky;
  d.pool_k = p.pool_k;
  d.pool_s = p.pool_s;
  d.nseg = in.nseg;
  const Segment& s0 = in.seg[0];
  d.seg0_off = i32(s0.off, "seg0_off");
  d.seg0_c = s0.c;
  d.seg0_h = s0.h;
  d.seg0_w = s0.w;
  d.seg0_up = s0.up;
  if (in.nseg > 1) {
    const Segment& s1 = in.seg[1];
    d.seg1_off = i32(s1.off, "seg1_off");
    d.seg1_c = s1.c;
    d.seg1_h = s1.h;
    d.seg1_w = s1.w;
    d.seg1_up = s1.up;
  }
  d.out_off = i32(out_off, "out_off");
  d.prepool_off = prepool_off < 0 ? -1 : i32(prepool_off, "prepool_off");
  if (w_off % accel::WORD) throw std::runtime_error("poids non alignés sur un mot");
  d.w_off = i32(w_off, "w_off");
  d.b_off = i32(b_index, "b_off");
  d.m0_off = i32(m0_index, "m0_off");
  // T10.4 : tuile agrandie pour les convs suivies d'un maxpool de stride 2 (tuile entière
  // utile), pliage de la conv d'entrée (cin·k ≤ Tn) ; sinon tuile Tr × Tc.
  const bool big = accel::TILE_POOL > 0 && p.pooled() && p.pool_s == 2;
  d.tr = big ? accel::TILE_POOL : accel::TR;
  d.tc = big ? accel::TILE_POOL : accel::TC;
  d.fold = accel::FOLD && p.k > 1 && p.cin * p.k <= accel::TN;
  d.wbits = l.wbits;  // T10.10 : poids paquetés par reorder_weights
  return d;
}

Program build(const golden::Model& m) {
  Program p;
  for (const auto& kv : m.buffers) {
    p.base[kv.first] = p.arena_size;
    p.arena_size += (kv.second + ARENA_ALIGN - 1) / ARENA_ALIGN * ARENA_ALIGN;
  }
  auto addr = [&](const golden::BufRef& r) { return p.base.at(r.buf) + r.offset; };
  p.arena_size += ARENA_SLACK;
  p.input_off = addr(m.input);
  p.params = m.bias;
  p.params.insert(p.params.end(), m.m0.begin(), m.m0.end());
  const int64_t m0_base = int64_t(m.bias.size());

  // Vues : même logique que golden::Engine::place_views (cpp/golden/src/engine.cpp).
  p.views.assign(m.layers.size(), View{});
  for (const Layer& l : m.layers) {
    View& v = p.views[size_t(l.id)];
    switch (l.type) {
      case LayerType::Conv: {
        const Layer* dst = l.pool_layer >= 0 ? &m.layers[size_t(l.pool_layer)] : &l;
        v = single_view(addr(l.out), dst->out_c, dst->out_h, dst->out_w);
        if (l.pool_layer >= 0) {
          p.views[size_t(l.pool_layer)] = v;
          v = l.prepool_out.valid() ? single_view(addr(l.prepool_out), l.cout,
                                                  golden::prepool_h(l), golden::prepool_w(l))
                                    : View{};
        }
        break;
      }
      case LayerType::Maxpool:
        if (l.fused_into < 0) throw std::runtime_error("maxpool non fusionné");
        break;  // posée par sa conv
      case LayerType::Upsample:
        v = p.views[size_t(l.id - 1)];
        if (v.nseg != 1 || l.s != 2) throw std::runtime_error("upsample ×2 d'une vue simple");
        v.seg[0].up += 1;
        v.h *= 2;
        v.w *= 2;
        break;
      case LayerType::Route:
        for (int j : l.from) {
          const View& s = p.views[size_t(j)];
          if (s.nseg == 0) throw std::runtime_error("source de route sans sortie en DDR");
          if (v.nseg > 0 && (s.h != v.h || s.w != v.w))
            throw std::runtime_error("route : formes incompatibles");
          v.h = s.h;
          v.w = s.w;
          for (int k = 0; k < s.nseg; ++k) {
            const Segment& g = s.seg[k];
            Segment* last = v.nseg ? &v.seg[v.nseg - 1] : nullptr;
            if (last && last->up == 0 && g.up == 0 && last->h == g.h && last->w == g.w &&
                last->off + int64_t(last->c) * last->h * last->w == g.off) {
              last->c += g.c;  // placement contigu : concaténation gratuite
            } else {
              if (v.nseg == accel::MAX_SEGMENTS)
                throw std::runtime_error("route : trop de segments");
              v.seg[v.nseg++] = g;
            }
            v.c += g.c;
          }
        }
        break;
      case LayerType::Yolo:
      case LayerType::Region:
        v = p.views[size_t(l.id - 1)];
        break;
    }
  }

  for (const Layer& l : m.layers) {
    if (l.type != LayerType::Conv) continue;
    const View in = l.id == 0 ? single_view(p.input_off, m.in_c, m.in_h, m.in_w)
                              : p.views[size_t(l.id - 1)];
    const int64_t pre = l.prepool_out.valid() ? addr(l.prepool_out) : -1;
    const int64_t w_off = int64_t(p.weights.size());
    accel::LayerDesc d =
        conv_desc(l, in, addr(l.out), pre, l.b_offset / 4, m0_base + l.m0_offset / 4, w_off);
    accel::reorder_weights(d, m.conv_weights(l), p.weights);
    p.w_off[l.id] = w_off;
    p.calls.push_back({l.id, d});
  }
  return p;
}

std::vector<int32_t> desc_table(const Program& p) {
  std::vector<int32_t> t(p.calls.size() * accel::D_WORDS);
  for (size_t i = 0; i < p.calls.size(); ++i)
    std::memcpy(&t[i * accel::D_WORDS], &p.calls[i].desc, sizeof(accel::LayerDesc));
  return t;
}

void run(const Program& p, const golden::Model& m, std::vector<Word>& arena,
         const int8_t* input, Kernel kernel, const std::function<void(const ConvCall&)>& after) {
  arena.assign(size_t(p.arena_size + accel::WORD - 1) / accel::WORD, 0);
  std::memcpy(reinterpret_cast<int8_t*>(arena.data()) + p.input_off, input,
              size_t(m.in_c) * m.in_h * m.in_w);
  // Poids réordonnés, copiés dans des mots (alignement du port m_axi).
  std::vector<Word> wts((p.weights.size() + accel::WORD - 1) / accel::WORD, 0);
  std::memcpy(wts.data(), p.weights.data(), p.weights.size());
  for (const ConvCall& c : p.calls) {
    kernel(arena.data(), arena.data(), wts.data(), p.params.data(), c.desc, nullptr, 0);
    if (after) after(c);
  }
}

}  // namespace driver
