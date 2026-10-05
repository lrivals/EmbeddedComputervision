// T5.2 : convolution tuilée == implémentation naïve et == dumps Python, pour plusieurs tuiles.
#include <catch2/catch_test_macros.hpp>
#include <random>

#include "common.hpp"
#include "golden/conv.hpp"
#include "golden/engine.hpp"

using namespace golden;

// Tailles des tampons == formules du §10.2 (vérifiées aussi par static_assert).
TEST_CASE("tailles des tampons == B_in, B_w, B_out") {
  using T = Tiles<16, 16, 13, 13>;
  REQUIRE(sizeof(T::Buffers::in_buf) == 16 * 15 * 15);
  REQUIRE(sizeof(T::Buffers::w_buf) == 16 * 16 * 9);
  REQUIRE(sizeof(T::Buffers::out_buf) / sizeof(int32_t) == 16 * 13 * 13);
  using U = Tiles<7, 5, 6, 9>;
  REQUIRE(sizeof(U::Buffers::in_buf) == buf_in_size(5, 6, 9, 3, 1));
  REQUIRE(buf_in_size(5, 6, 9, 3, 1) == 5 * 8 * 11);
  REQUIRE(buf_in_size(4, 3, 3, 3, 2) == 4 * 7 * 7);  // forme générale S Tr + K − S
}

namespace {

// Référence naïve (contrat de conventions.md « Arithmétique entière »).
std::vector<int8_t> naive(const ConvParams& p, const std::vector<int8_t>& x,
                          const std::vector<int8_t>& w, const std::vector<int32_t>& b,
                          const std::vector<int32_t>& m0, std::vector<int8_t>* pre) {
  const int R = p.out_h(), C = p.out_w(), K = p.k;
  std::vector<int8_t> y(size_t(p.cout) * R * C);
  for (int o = 0; o < p.cout; ++o)
    for (int r = 0; r < R; ++r)
      for (int c = 0; c < C; ++c) {
        int64_t acc = b[size_t(o)];
        for (int i = 0; i < p.cin; ++i)
          for (int u = 0; u < K; ++u)
            for (int v = 0; v < K; ++v) {
              const int ir = r - p.pad + u, ic = c - p.pad + v;
              if (ir < 0 || ir >= p.h || ic < 0 || ic >= p.w) continue;
              acc += int64_t(w[((size_t(o) * p.cin + i) * K + u) * K + v]) *
                     x[(size_t(i) * p.h + ir) * p.w + ic];
            }
        int32_t q = requantize(int32_t(acc), m0[size_t(o)], p.shift);
        if (p.leaky) q = leaky_int(q);
        y[(size_t(o) * R + r) * C + c] = clip8(q);
      }
  if (pre) *pre = y;
  if (!p.pooled()) return y;
  const int Rp = p.pool_h(), Cp = p.pool_w(), k = p.pool_k, s = p.pool_s;
  std::vector<int8_t> z(size_t(p.cout) * Rp * Cp);
  for (int o = 0; o < p.cout; ++o)
    for (int r = 0; r < Rp; ++r)
      for (int c = 0; c < Cp; ++c) {
        int m = -128;
        for (int u = 0; u < k; ++u)
          for (int v = 0; v < k; ++v) {
            const int rr = std::min(r * s + u, R - 1), cc = std::min(c * s + v, C - 1);
            m = std::max(m, int(y[(size_t(o) * R + rr) * C + cc]));
          }
        z[(size_t(o) * Rp + r) * Cp + c] = int8_t(m);
      }
  return z;
}

template <class T>
void check_synthetic(const ConvParams& p, uint32_t seed) {
  std::mt19937 g(seed);
  std::uniform_int_distribution<int> q8(-127, 127), qb(-5000, 5000);
  std::uniform_int_distribution<int32_t> qm(1 << 29, (1u << 31) - 1);
  std::vector<int8_t> x(size_t(p.cin) * p.h * p.w), w(size_t(p.cout) * p.cin * p.k * p.k);
  std::vector<int32_t> b(size_t(p.cout)), m0(size_t(p.cout));
  for (auto& v : x) v = int8_t(q8(g));
  for (auto& v : w) v = int8_t(q8(g));
  for (auto& v : b) v = qb(g);
  for (auto& v : m0) v = qm(g);
  std::vector<int8_t> pre_ref;
  const auto ref = naive(p, x, w, b, m0, &pre_ref);
  std::vector<int8_t> out(ref.size()), pre(pre_ref.size()), tap(pre_ref.size());
  conv_layer<T>(p, single_view(x.data(), p.cin, p.h, p.w), w.data(), b.data(), m0.data(),
                out.data(), p.pooled() ? pre.data() : nullptr, p.pooled() ? tap.data() : nullptr);
  REQUIRE(out == ref);
  if (p.pooled()) {
    REQUIRE(pre == pre_ref);
    REQUIRE(tap == pre_ref);
  }
}

ConvParams synth(int k, int cin, int cout, int h, int w, int pool_s, bool leaky) {
  ConvParams p;
  p.k = k;
  p.pad = (k - 1) / 2;
  p.cin = cin;
  p.cout = cout;
  p.h = h;
  p.w = w;
  p.shift = 38 - 8;  // M0 ~ 2³⁰, acc ~ 2¹⁶ : sorties de l'ordre de ±2⁶
  p.leaky = leaky;
  if (pool_s) {
    p.pool_k = 2;
    p.pool_s = pool_s;
  }
  return p;
}

template <class T>
void check_all_synthetic() {
  uint32_t seed = 1;
  for (int k : {1, 3})
    for (int pool : {0, 1, 2}) {
      const int h = pool == 2 ? 10 : 9, w = pool == 2 ? 14 : 11;  // s2 : dimensions paires
      check_synthetic<T>(synth(k, 5, 7, h, w, pool, true), seed++);
      check_synthetic<T>(synth(k, 19, 3, h, w, pool, k == 1), seed++);
    }
}

}  // namespace

TEST_CASE("conv tuilée == référence naïve (pad, 1×1, pool s2 et s1 au bord)") {
  check_all_synthetic<Tiles<16, 16, 13, 13>>();
  check_all_synthetic<Tiles<4, 3, 5, 4>>();
  check_all_synthetic<Tiles<1, 1, 2, 2>>();
  check_all_synthetic<Tiles<7, 5, 6, 9>>();
}

namespace {

// Chaque conv seule : entrée = dump de la couche précédente, sorties == dumps.
template <class T>
void check_layers(const char* net) {
  const Model& m = testing::model(net);
  for (const Layer& l : m.layers) {
    if (l.type != LayerType::Conv) continue;
    INFO(net << " couche " << l.id);
    const NpyInt8 x = testing::dump(net, "000001", l.id - 1);
    const ConvParams p = conv_params(l, int(x.shape[2]), int(x.shape[3]));
    REQUIRE(x.shape[1] == p.cin);
    const NpyInt8 want = testing::dump(net, "000001", p.pooled() ? l.pool_layer : l.id);
    std::vector<int8_t> out(want.numel()), tap;
    if (p.pooled()) tap.resize(size_t(p.cout) * p.out_h() * p.out_w());
    conv_layer<T>(p, single_view(x.data.data(), p.cin, p.h, p.w), m.conv_weights(l),
                  m.conv_bias(l), m.conv_m0(l), out.data(), nullptr,
                  p.pooled() ? tap.data() : nullptr);
    CHECK(out == want.data);
    if (p.pooled()) CHECK(tap == testing::dump(net, "000001", l.id).data);
  }
}

}  // namespace

TEST_CASE("chaque conv == dump Python, plusieurs jeux de tuiles") {
  for (const char* net : testing::NETS) {
    REQUIRE_MODEL(net);
    SECTION(std::string(net) + " 16,16,13,13") { check_layers<Tiles<16, 16, 13, 13>>(net); }
    SECTION(std::string(net) + " 7,5,6,9 (non diviseurs)") { check_layers<Tiles<7, 5, 6, 9>>(net); }
    SECTION(std::string(net) + " 32,8,26,4") { check_layers<Tiles<32, 8, 26, 4>>(net); }
  }
}
