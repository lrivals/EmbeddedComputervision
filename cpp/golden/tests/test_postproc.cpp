// T5.5 : post-traitement C++ == detections.json (égalité exacte des doubles).
#include <catch2/catch_test_macros.hpp>

#include "common.hpp"
#include "golden/json.hpp"
#include "golden/postproc.hpp"

using namespace golden;

TEST_CASE("seuil sur le logit") {
  // σ(q/8) > 0,25 ⟺ q ≥ ⌊8 ln(1/3)⌋ + 1 = −8
  REQUIRE(postproc::logit_threshold_q(0.25, 0.125) == -8);
  REQUIRE(postproc::logit_threshold_q(0.5, 0.125) == 1);
}

TEST_CASE("NMS : suppression au-delà du seuil, tri stable") {
  const double a[4] = {0.5, 0.5, 0.2, 0.2}, b[4] = {0.51, 0.5, 0.2, 0.2},
               c[4] = {0.1, 0.1, 0.1, 0.1};
  const auto keep = postproc::nms({a, b, c}, {0.5, 0.9, 0.5}, 0.45);
  REQUIRE(keep == std::vector<size_t>{1, 2});
}

// Les têtes viennent des dumps Python : le post-traitement est testé seul.
TEST_CASE("détections == detections.json") {
  for (const char* net : testing::NETS) {
    REQUIRE_MODEL(net);
    const Model& m = testing::model(net);
    for (const char* image : testing::IMAGES) {
      INFO(net << " image " << image);
      std::vector<NpyInt8> data;
      std::vector<postproc::Head> heads;
      for (int id : m.heads()) data.push_back(testing::dump(net, image, id));
      size_t k = 0;
      for (int id : m.heads()) {
        const Layer& l = m.layers[size_t(id)];
        postproc::Head h;
        h.data = data[k++].data.data();
        h.grid_h = l.out_h;
        h.grid_w = l.out_w;
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
        heads.push_back(h);
      }
      const Json ref = Json::parse_file(testing::model_dir(net) + "/dumps/" + image +
                                        "/detections.json");
      const auto dets = postproc::postprocess(heads, ref["conf"].num(), ref["iou"].num());
      REQUIRE(dets.size() == ref["labels"].size());
      for (size_t d = 0; d < dets.size(); ++d) {
        for (int c = 0; c < 4; ++c) REQUIRE(dets[d].box[c] == ref["boxes"][d][size_t(c)].num());
        REQUIRE(dets[d].score == ref["scores"][d].num());
        REQUIRE(dets[d].label == ref["labels"][d].integer());
      }
    }
  }
}
