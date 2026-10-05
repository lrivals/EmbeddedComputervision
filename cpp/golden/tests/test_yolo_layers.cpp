// T5.3 : maxpool 2×2/1, upsample par division d'adresse, route par placement contigu.
#include <catch2/catch_test_macros.hpp>
#include <numeric>

#include "common.hpp"
#include "golden/engine.hpp"

using namespace golden;

namespace {

void require_dump(const Engine& e, const std::string& net, int id) {
  INFO(net << " couche " << id);
  const Tensor& t = e.recorded()[size_t(id)];
  const NpyInt8 want = testing::dump(net, "000001", id);
  REQUIRE(want.shape == std::vector<int64_t>{1, t.c, t.h, t.w});
  REQUIRE(t.data == want.data);
}

}  // namespace

TEST_CASE("Tiny-YOLOv2 : maxpool stride 1 (couche 11)") {
  const char* net = "tiny-yolov2-voc";
  REQUIRE_MODEL(net);
  Engine e(testing::model(net));
  e.run(testing::dump(net, "000001", -1).data.data(), true);
  require_dump(e, net, 10);
  require_dump(e, net, 11);
}

TEST_CASE("Tiny-YOLOv3 : couches 11, 19, 20 sans copie mémoire") {
  const char* net = "tiny-yolov3-coco";
  REQUIRE_MODEL(net);
  const Model& m = testing::model(net);
  Engine e(m);
  // Témoin : la zone R:[0, 128·26·26) réservée par le manifest à une copie de l'upsample.
  const Layer& l8 = m.layers[8];
  const size_t up_bytes = size_t(l8.prepool_out.offset);
  REQUIRE(up_bytes == 128u * 26 * 26);
  std::fill(e.buffer("R").begin(), e.buffer("R").begin() + long(up_bytes), int8_t(0x5A));

  e.run(testing::dump(net, "000001", -1).data.data(), true);
  for (int id : {11, 19, 20}) require_dump(e, net, id);

  // Adressage : upsample = division d'adresse de la sortie de L18, route = deux segments
  // (L18 vue ×2 puis prépool de L08) ; route 17 = la carte de L13 en place.
  const InView& v19 = e.view(19);
  REQUIRE(v19.nseg == 1);
  REQUIRE(v19.seg[0].up == 1);
  REQUIRE(v19.seg[0].base == e.buffer("B").data());
  const InView& v20 = e.view(20);
  REQUIRE(v20.nseg == 2);
  REQUIRE(v20.seg[1].base == e.buffer("R").data() + l8.prepool_out.offset);
  REQUIRE(e.view(17).seg[0].base == e.buffer("S").data());

  // Compteurs d'accès : aucune lecture ni écriture pour upsample et routes.
  for (int id : {17, 19, 20}) {
    INFO("couche " << id);
    REQUIRE(e.stats(id).act_rd == 0);
    REQUIRE(e.stats(id).act_wr == 0);
    REQUIRE(e.stats(id).param_rd == 0);
  }
  // Octets écrits par le réseau = sorties des convs (+ prépool de L08), rien d'autre.
  uint64_t written = 0, expected = 0;
  for (const Layer& l : m.layers) {
    written += e.stats(l.id).act_wr;
    if (l.type != LayerType::Conv) continue;
    expected += uint64_t(l.out_c) * l.out_h * l.out_w;
    if (l.prepool_out.valid()) expected += uint64_t(l.cout) * prepool_h(l) * prepool_w(l);
  }
  REQUIRE(written == expected);
  const auto& R = e.buffer("R");
  REQUIRE(std::all_of(R.begin(), R.begin() + long(up_bytes), [](int8_t v) { return v == 0x5A; }));
}
