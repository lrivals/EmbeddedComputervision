// T25.2: ABFT checksum of every conv. Fault-free: the check passes on every conv of the dumps
// (several tile sizes); one corrupted weight: the check fails, as the outputs change.
#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include "common.hpp"
#include "golden/conv.hpp"
#include "golden/engine.hpp"

using namespace golden;

namespace {

template <class T>
AccessStats run_conv(const Model& m, const Layer& l, const NpyInt8& x, const int8_t* w,
                     std::vector<int8_t>& out) {
  ConvParams p = conv_params(l, int(x.shape[2]), int(x.shape[3]));
  p.cs_w = m.conv_checksum(l);
  p.cs_b = l.cs_b;
  out.assign(size_t(p.cout) * p.pool_h() * p.pool_w(), 0);
  AccessStats st;
  conv_layer<T>(p, single_view(x.data.data(), p.cin, p.h, p.w), w, m.conv_bias(l),
                m.conv_m0(l), out.data(), nullptr, nullptr, &st);
  return st;
}

template <class T>
void check_fault_free(const char* net) {
  const Model& m = testing::model(net);
  for (const char* image : testing::IMAGES)
    for (const Layer& l : m.layers) {
      if (l.type != LayerType::Conv) continue;
      INFO(net << " image " << image << " conv " << l.id);
      REQUIRE(m.conv_checksum(l) != nullptr);
      const NpyInt8 x = testing::dump(net, image, l.id - 1);
      std::vector<int8_t> out;
      const AccessStats st = run_conv<T>(m, l, x, m.conv_weights(l), out);
      CHECK(st.abft_checks > 0);
      CHECK(st.abft_errors == 0);
      const NpyInt8 want = testing::dump(net, image, l.pool_layer >= 0 ? l.pool_layer : l.id);
      CHECK(out == want.data);  // outputs unchanged by the check
    }
}

}  // namespace

TEST_CASE("T25.2: ABFT passes on every conv of the dumps") {
  for (const char* net : testing::NETS) {
    REQUIRE_MODEL(net);
    SECTION(std::string(net) + " 16,16,13,13") { check_fault_free<Tiles<16, 16, 13, 13>>(net); }
    SECTION(std::string(net) + " 7,5,6,9") { check_fault_free<Tiles<7, 5, 6, 9>>(net); }
  }
}

TEST_CASE("T25.2: ABFT detects a corrupted weight") {
  const char* net = testing::NETS[0];
  REQUIRE_MODEL(net);
  const Model& m = testing::model(net);
  int detected = 0;
  for (const Layer& l : m.layers) {
    if (l.type != LayerType::Conv) continue;
    INFO("conv " << l.id);
    const NpyInt8 x = testing::dump(net, "000001", l.id - 1);
    const size_t n = size_t(l.cout) * l.cin * l.k * l.k;
    std::vector<int8_t> w(m.conv_weights(l), m.conv_weights(l) + n);
    const size_t at = n / 2 + size_t(l.k * l.k / 2);  // a centre tap: it meets the image
    w[at] = int8_t(w[at] ^ 0x40);                      // bit 6 flipped
    std::vector<int8_t> ref, out;
    run_conv<DefaultTiles>(m, l, x, m.conv_weights(l), ref);
    const AccessStats st = run_conv<DefaultTiles>(m, l, x, w.data(), out);
    // One filter changed: its accumulator changes where the tap meets x ≠ 0, so does the
    // sum, exactly. An output that changed is always detected.
    if (out != ref) CHECK(st.abft_errors > 0);
    detected += st.abft_errors > 0;
  }
  CHECK(detected > 0);
}

TEST_CASE("T25.2: the engine counts no ABFT error on the dumps") {
  for (const char* net : testing::NETS) {
    REQUIRE_MODEL(net);
    const Model& m = testing::model(net);
    Engine e(m);
    e.run(testing::dump(net, "000002", -1).data.data());
    CHECK(e.abft_errors() == 0);
    uint64_t checks = 0;
    for (const Layer& l : m.layers) checks += e.stats(l.id).abft_checks;
    CHECK(checks > 0);
  }
}
