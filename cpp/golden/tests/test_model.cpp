// T5.1 : lecture du manifest, des blobs et des dumps.
#include <catch2/catch_test_macros.hpp>

#include "common.hpp"
#include "golden/json.hpp"

using namespace golden;

TEST_CASE("JSON : nombres en aller-retour exact, imbrication") {
  const Json j = Json::parse(R"({"a": [1, -2.5e-3, 0.338038316861851], "b": {"c": "xé"},
                                 "n": null, "t": true})");
  REQUIRE(j["a"][0].integer() == 1);
  REQUIRE(j["a"][1].num() == -2.5e-3);
  REQUIRE(j["a"][2].num() == 0.338038316861851);
  REQUIRE(j["b"]["c"].str() == "x\xc3\xa9");
  REQUIRE(j["n"].is_null());
  REQUIRE(j["t"].boolean());
  REQUIRE_THROWS(Json::parse("{\"a\": }"));
}

TEST_CASE("npy : écriture puis relecture") {
  const int8_t data[6] = {-127, -1, 0, 1, 2, 127};
  const std::string path = std::filesystem::temp_directory_path() / "golden_npy_test.npy";
  npy_save_int8(path, {1, 2, 3}, data);
  const NpyInt8 a = npy_load_int8(path);
  REQUIRE(a.shape == std::vector<int64_t>{1, 2, 3});
  REQUIRE(std::vector<int8_t>(data, data + 6) == a.data);
}

TEST_CASE("manifest et blobs des exports réels") {
  for (const char* net : testing::NETS) {
    REQUIRE_MODEL(net);
    const Model& m = testing::model(net);
    INFO(net);
    REQUIRE(m.network == net);
    REQUIRE(m.in_c == 3);
    REQUIRE(m.input_scale == 1.0 / 127);
    const bool v3 = m.network == "tiny-yolov3-coco";
    REQUIRE(m.layers.size() == (v3 ? 24u : 16u));
    REQUIRE(m.heads() == (v3 ? std::vector<int>{16, 23} : std::vector<int>{15}));

    // Poids contigus et complets ; chaînage des échelles le long des convs.
    int64_t w_end = 0;
    double prev_scale = m.input_scale;
    for (const Layer& l : m.layers) {
      if (l.type == LayerType::Conv) {
        REQUIRE(l.w_offset >= w_end);
        w_end = l.w_offset + int64_t(l.cout) * l.cin * l.k * l.k;
        REQUIRE(l.in_scale == prev_scale);
        prev_scale = l.out_scale;
        if (l.pool_layer >= 0) REQUIRE(m.layers[size_t(l.pool_layer)].fused_into == l.id);
      } else if (l.type == LayerType::Route || l.type == LayerType::Upsample) {
        prev_scale = l.scale;  // route et upsample conservent l'échelle de leur entrée
      }
    }
    REQUIRE(w_end <= int64_t(m.weights.size()));
    REQUIRE(int64_t(m.weights.size()) - w_end < BLOB_ALIGN);

    // Dumps lisibles et de la forme du manifest.
    const NpyInt8 x = testing::dump(net, "000001", -1);
    REQUIRE(x.shape == std::vector<int64_t>{1, m.in_c, m.in_h, m.in_w});
    const Layer& last = m.layers.back();
    REQUIRE(testing::dump(net, "000001", last.id).shape ==
            std::vector<int64_t>{1, last.out_c, last.out_h, last.out_w});
  }
}
