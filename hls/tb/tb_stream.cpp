// Testbench de l'architecture streaming (T9.4.2, T10.8) : sortie de chaque étage et tête de
// `yolo_stream` == dumps Python (== golden) ; cycles par étage (itérations PE × SIMD) ; tête
// AXI-Stream avec un seul TLAST. Compilé avec et sans STREAM_ROM (tb_stream_rom).
//
//   tb_stream [--model DIR] [--image ID]…   (Tiny-YOLOv2 VOC seulement)
#include <cstdio>

#include "golden/model.hpp"
#include "tb_common.hpp"
#include "yolo_stream.hpp"

using namespace golden;

int main(int argc, char** argv) {
  tb::Args a = tb::parse(argc, argv);
  const std::string net = "tiny-yolov2-voc";
  if (!tb::have_model(a, net)) {
    std::printf("%s : export absent (make export)\n", net.c_str());
    return 1;
  }
  const Model m = Model::load(tb::net_dir(a, net));
  stream::StreamDesc d{};
  for (int k = 0; k < stream::N_STAGES; ++k) {
    const Layer& l = m.layers[size_t(stream::STAGE_LAYER[k])];
    if (l.type != LayerType::Conv) throw std::runtime_error("étage : pas une conv");
    d.w_off[k] = int32_t(l.w_offset);
    d.b_off[k] = int32_t(l.b_offset / 4);
    d.m0_off[k] = int32_t(l.m0_offset / 4);
    d.shift[k] = l.shift;
    d.qmax[k] = l.qmax;
  }
  stream::stream_tap = true;
  size_t total = 0;
  for (const std::string& image : a.images) {
    const NpyInt8 x = tb::dump(a, net, image, -1);
    const int C = m.in_c, H = m.in_h, W = m.in_w;
    hls::stream<axis_byte> in, out;
    for (int r = 0; r < H; ++r)  // CHW → HWC
      for (int c = 0; c < W; ++c)
        for (int ch = 0; ch < C; ++ch) in.write({x.data[(size_t(ch) * H + r) * W + c], false});
    yolo_stream(in, out, m.weights.data(), m.bias.data(), m.m0.data(), d);
    // Tête en AXI-Stream : TLAST sur la dernière valeur seulement, == tap de l'étage 8.
    std::vector<int8_t> head;
    size_t n_last = 0;
    while (!out.empty()) {
      const axis_byte v = out.read();
      head.push_back(v.data);
      n_last += v.last;
    }
    const bool axis_ok = head == stream::stream_taps[stream::N_STAGES - 1] && n_last == 1 &&
                         head.size() == size_t(stream::OUT_VALUES);
    if (!axis_ok) std::printf("  ÉCART sortie AXI-Stream (valeurs ou TLAST)\n");
    size_t nd_img = 0;
    uint64_t ii = 0;
    for (int k = 0; k < stream::N_STAGES; ++k) {
      // Sortie de l'étage k = carte poolée (couche suivante) ou sortie conv.
      const Layer& l = m.layers[size_t(stream::STAGE_LAYER[k])];
      const int id = l.pool_layer >= 0 ? l.pool_layer : l.id;
      const NpyInt8 want = tb::dump(a, net, image, id);
      const std::vector<int8_t>& got = stream::stream_taps[k];
      const int oc = l.out_c, oh = l.out_h, ow = l.out_w;
      std::vector<int8_t> chw(got);  // STAGE_OUT_CHW : déjà en CHW
      if (!stream::STAGE_OUT_CHW[k] && got.size() == size_t(oc) * oh * ow)
        for (int r = 0; r < oh; ++r)
          for (int c = 0; c < ow; ++c)
            for (int ch = 0; ch < oc; ++ch)
              chw[(size_t(ch) * oh + r) * ow + c] = got[(size_t(r) * ow + c) * oc + ch];
      char what[48];
      std::snprintf(what, sizeof what, "étage %d (L%02d)", k, id);
      size_t nd = 0;
      if (got.size() != want.data.size()) {
        std::printf("  %s : %zu valeurs au lieu de %zu\n", what, got.size(), want.data.size());
        nd = 1;
      } else {
        nd = tb::compare(what, chw.data(), want.data.data(), chw.size());
      }
      nd_img += nd;
      const uint64_t cyc = stream::stream_cycles[k].mac;
      ii = cyc > ii ? cyc : ii;
      std::printf("  %s : PE %d × SIMD %d, %llu cycles, %s\n", what, stream::STAGE_PE[k],
                  stream::STAGE_SIMD[k], (unsigned long long)cyc, nd ? "ÉCHEC" : "OK");
      stream::stream_cycles[k] = stream::StageCycles{};
    }
    nd_img += axis_ok ? 0 : 1;
    std::printf("%s : %s, II = %llu cycles (étage le plus lent)\n", image.c_str(),
                nd_img ? "ÉCHEC" : "OK", (unsigned long long)ii);
    total += nd_img;
  }
  std::printf("%zu écarts\n", total);
  return total == 0 ? 0 : 1;
}
