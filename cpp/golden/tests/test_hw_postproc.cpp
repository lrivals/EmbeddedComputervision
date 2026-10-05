// T9.1.1 : primitives du post-traitement matériel (l'égalité à Python sur les dumps et des
// têtes aléatoires est testée par python/tests/test_hw_postproc.py via `golden_run hwpp`).
#include <catch2/catch_test_macros.hpp>

#include "common.hpp"
#include "golden/hw_heads.hpp"

using namespace golden;

static hwpp::Box box(int x, int score, int cls = 0, int w = 100) {
  return hwpp::Box{x, 0, x + w, w, score, cls};
}

TEST_CASE("IoU > 9/20 par produits croisés") {
  int p = 0, q = 0;
  hwpp::iou_fraction(0.45, p, q);
  REQUIRE((p == 9 && q == 20));
  // 100×100 décalées de d : IoU = (100 − d)/(100 + d) > 0,45 ⟺ d < 37,9
  REQUIRE(hwpp::overlaps(box(0, 1), box(37, 1), p, q));
  REQUIRE_FALSE(hwpp::overlaps(box(0, 1), box(38, 1), p, q));
}

TEST_CASE("NMS sans tri : remplacement, rejet, chaîne, débordement") {
  hwpp::Selector<4> s;
  s.push(box(0, 100), 9, 20);
  s.push(box(10, 200), 9, 20);     // remplace la première
  s.push(box(5, 50, 1), 9, 20);    // autre classe
  s.push(box(12, 200), 9, 20);     // égalité : rejetée
  REQUIRE(s.n == 2);
  REQUIRE((s.sel[0].x1 == 10 && s.sel[0].score == 200 && s.alive[0]));
  REQUIRE(s.sel[1].cls == 1);
  hwpp::Selector<2> t;
  for (int k = 0; k < 4; ++k) t.push(box(1000 * k, 10), 9, 20);
  REQUIRE((t.n == 2 && t.overflow == 2));
}

TEST_CASE("descripteurs de têtes") {
  for (const char* net : testing::NETS) {
    REQUIRE_MODEL(net);
    const Model& m = testing::model(net);
    for (int id : m.heads()) {
      const Layer& l = m.layers[size_t(id)];
      const hwpp::HeadData hd = make_hw_head(m, l, nullptr, 0.25);
      REQUIRE((1 << hd.desc.stride_log2) * hd.desc.grid == 416);
      REQUIRE(hd.desc.obj_thr_q == -8);  // σ(q/8) > 0,25 ⟺ q ≥ −8 (têtes au pas 1/8)
    }
  }
}
