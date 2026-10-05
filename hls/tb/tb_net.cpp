// Testbench réseau complet (T6.4, T6.5) : le driver enchaîne les appels du noyau sur une
// arène DDR contiguë, upsample et route par adressage seulement. Chaque couche présente en
// DDR == golden::Engine (couche par couche) == dump Python.
//
//   tb_net [--model DIR] [--net NAME]… [--image ID]… [--csv cycles.csv]
//
// En co-sim RTL (cosim.tcl), on le restreint à une image : `--net tiny-yolov2-voc --image 000001`.
#include <cstdio>
#include <fstream>

#include "accel.hpp"
#include "golden/engine.hpp"
#include "golden/model.hpp"
#include "program.hpp"
#include "tb_common.hpp"

using namespace golden;
using Tl = Tiles<accel::TM, accel::TN, accel::TR, accel::TC>;

int main(int argc, char** argv) {
  const tb::Args a = tb::parse(argc, argv);
  std::printf("tuiles Tm %d Tn %d Tr %d Tc %d\n", accel::TM, accel::TN, accel::TR, accel::TC);
  std::ofstream csv;
  if (!a.csv.empty()) {
    csv.open(a.csv);
    csv << "net,image,layer,overlapped,sequential,compute\n";
  }
  size_t total_diff = 0;
  int checked = 0;
  for (const std::string& net : a.nets) {
    if (!tb::have_model(a, net)) {
      std::printf("%s : export absent (make export), sauté\n", net.c_str());
      continue;
    }
    const Model m = Model::load(tb::net_dir(a, net));
    const driver::Program prog = driver::build(m);
    std::printf("%s : arène %lld octets, %zu appels du noyau\n", net.c_str(),
                (long long)prog.arena_size, prog.calls.size());
    Engine ref(m);

    for (const std::string& image : a.images) {
      const NpyInt8 x = tb::dump(a, net, image, -1);
      ref.run<Tl>(x.data.data(), true);

      // Vérification juste après chaque appel : les tampons ping-pong A/B sont réécrits
      // par les couches suivantes. Après la conv L, on contrôle les couches L … (conv
      // suivante − 1) : la conv, son maxpool fusionné, puis les vues sans calcul (upsample,
      // route, têtes) tant que leurs sources sont en place — comme Engine::record.
      std::vector<driver::Word> arena;
      uint64_t cycles = 0;
      size_t nd_img = 0;
      size_t next = 1;
      driver::run(prog, m, arena, x.data.data(), yolo_conv, [&](const driver::ConvCall& c) {
        const accel::SimCycles& s = accel::sim_cycles;
        cycles += s.overlapped;
        if (csv.is_open())
          csv << net << ',' << image << ',' << c.layer << ',' << s.overlapped << ','
              << s.sequential << ',' << s.compute << '\n';
        const int end = next < prog.calls.size() ? prog.calls[next].layer : int(m.layers.size());
        ++next;
        for (int id = c.layer; id < end; ++id) {
          const Layer& l = m.layers[size_t(id)];
          const driver::View& v = prog.views[size_t(id)];
          if (v.nseg == 0) continue;  // conv poolée sans carte avant pooling en DDR
          const std::vector<int8_t> got =
              driver::materialize(reinterpret_cast<const int8_t*>(arena.data()), v);
          const Tensor& g = ref.recorded()[size_t(id)];
          const NpyInt8 want = tb::dump(a, net, image, id);
          char what[48];
          std::snprintf(what, sizeof what, "L%02d %s vs golden", id, layer_type_name(l.type));
          if (g.data.size() != got.size() || want.data.size() != got.size()) {
            std::printf("  %s : taille %zu au lieu de %zu\n", what, got.size(), g.data.size());
            ++nd_img;
            continue;
          }
          nd_img += tb::compare(what, got.data(), g.data.data(), got.size());
          std::snprintf(what, sizeof what, "L%02d %s vs dump", id, layer_type_name(l.type));
          nd_img += tb::compare(what, got.data(), want.data.data(), got.size());
          ++checked;
        }
      });
      total_diff += nd_img;
      std::printf("%s %s : %s, ≈ %llu cycles (estimation C-sim)\n", net.c_str(), image.c_str(),
                  nd_img ? "ÉCHEC" : "OK", (unsigned long long)cycles);
    }
  }
  std::printf("%d sorties de couche vérifiées, %zu écarts\n", checked, total_diff);
  return (total_diff == 0 && checked > 0) ? 0 : 1;
}
