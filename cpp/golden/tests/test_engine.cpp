// T5.4 : réseau complet depuis input.npy, 0 valeur différente sur toutes les couches.
#include <catch2/catch_test_macros.hpp>

#include "common.hpp"
#include "golden/engine.hpp"

using namespace golden;

TEST_CASE("réseau complet == dumps Python (3 images, v2 et v3)") {
  for (const char* net : testing::NETS) {
    REQUIRE_MODEL(net);
    const Model& m = testing::model(net);
    Engine e(m);
    for (const char* image : testing::IMAGES) {
      e.run(testing::dump(net, image, -1).data.data(), true);
      for (const Layer& l : m.layers) {
        INFO(net << " image " << image << " couche " << l.id);
        const Tensor& t = e.recorded()[size_t(l.id)];
        const NpyInt8 want = testing::dump(net, image, l.id);
        REQUIRE(want.shape == std::vector<int64_t>{1, t.c, t.h, t.w});
        size_t ndiff = 0;
        for (size_t k = 0; k < t.data.size(); ++k) ndiff += t.data[k] != want.data[k];
        CHECK(ndiff == 0);
      }
    }
  }
}
