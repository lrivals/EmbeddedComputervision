// Bout-en-bout sur carte (T7.4) : chaque couche produite en DDR par l'accélérateur == dump
// golden, puis détections du post-traitement ARM == detections.json (T7.3).
//
//   run_compare [--model DIR] [--net NAME]… [--image ID]… [--layer N] [--csv temps.csv]
//               [--backend sim|uio] [--uio /dev/uioN] [--udmabuf udmabufN] [--poll]
//
// --layer N (T7.2, couche isolée) : l'arène est d'abord remplie avec les dumps des couches
// < N, aux places où les convs précédentes les auraient écrites, puis seule la conv N tourne.
// Code de sortie 0 = aucun écart.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include "accel_driver.hpp"
#include "golden/detections_io.hpp"
#include "golden/model.hpp"
#include "heads.hpp"
#include "options.hpp"
#include "tb_common.hpp"

using namespace golden;

namespace {

struct Args {
  tb::Args tb;
  driver::DeviceOptions dev;
};

Args parse(int argc, char** argv) {
  Args a;
  a.tb.model_dir = SW_MODEL_DIR;
  for (int i = 1; i < argc; ++i) {
    if (sw::parse_device_option(argc, argv, i, a.dev)) continue;
    const std::string k = argv[i];
    if (i + 1 >= argc) {
      std::fprintf(stderr, "argument inconnu ou sans valeur : %s\n", k.c_str());
      std::exit(2);
    }
    const std::string v = argv[++i];
    if (k == "--model") a.tb.model_dir = v;
    else if (k == "--net") a.tb.nets.push_back(v);
    else if (k == "--image") a.tb.images.push_back(v);
    else if (k == "--layer") a.tb.layer = std::atoi(v.c_str());
    else if (k == "--csv") a.tb.csv = v;
    else {
      std::fprintf(stderr, "argument inconnu : %s\n", k.c_str());
      std::exit(2);
    }
  }
  if (a.tb.nets.empty()) a.tb.nets.assign(std::begin(tb::ALL_NETS), std::end(tb::ALL_NETS));
  if (a.tb.images.empty())
    a.tb.images.assign(std::begin(tb::ALL_IMAGES), std::end(tb::ALL_IMAGES));
  return a;
}

// Écrit un dump (c, h, w) dans l'arène à la place d'une vue simple.
void put(int8_t* arena, const driver::View& v, const NpyInt8& x) {
  std::copy(x.data.begin(), x.data.end(), arena + v.seg[0].off);
}

bool plain(const driver::View& v) { return v.nseg == 1 && v.seg[0].up == 0; }

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args a = parse(argc, argv);
    std::printf("backend %s\n", a.dev.backend.c_str());
    std::ofstream csv;
    if (!a.tb.csv.empty()) {
      csv.open(a.tb.csv);
      csv << "backend,net,image,layer,seconds\n";
    }
    size_t total_diff = 0;
    int checked = 0;
    for (const std::string& net : a.tb.nets) {
      if (!tb::have_model(a.tb, net)) {
        std::printf("%s : export absent (make export), sauté\n", net.c_str());
        continue;
      }
      const Model m = Model::load(tb::net_dir(a.tb, net));
      // Un périphérique par réseau : en `uio`, le u-dma-buf entier va à un seul accélérateur.
      const auto dev = driver::make_device(a.dev);
      driver::Accelerator acc(*dev, m);
      const driver::Program& prog = acc.program();
      std::printf("%s : arène %lld octets, %zu appels du noyau\n", net.c_str(),
                  (long long)prog.arena_size, prog.calls.size());

      for (const std::string& image : a.tb.images) {
        const NpyInt8 x = tb::dump(a.tb, net, image, -1);
        size_t nd_img = 0;
        double t_acc = 0.0;

        // Après la conv `c` : couches [c, conv suivante) présentes en DDR == dump (même
        // boucle que hls/tb/tb_net.cpp).
        auto check = [&](size_t k) {
          const int first = prog.calls[k].layer;
          const int end = k + 1 < prog.calls.size() ? prog.calls[k + 1].layer
                                                    : int(m.layers.size());
          for (int id = first; id < end; ++id) {
            const driver::View& v = prog.views[size_t(id)];
            if (v.nseg == 0) continue;
            const std::vector<int8_t> got = driver::materialize(acc.arena(), v);
            const NpyInt8 want = tb::dump(a.tb, net, image, id);
            char what[48];
            std::snprintf(what, sizeof what, "L%02d %s", id,
                          layer_type_name(m.layers[size_t(id)].type));
            if (want.data.size() != got.size()) {
              std::printf("  %s : taille %zu au lieu de %zu\n", what, got.size(),
                          want.data.size());
              ++nd_img;
              continue;
            }
            nd_img += tb::compare(what, got.data(), want.data.data(), got.size());
            ++checked;
          }
        };
        auto log = [&](int layer, double s) {
          t_acc += s;
          if (csv.is_open())
            csv << dev->name() << ',' << net << ',' << image << ',' << layer << ',' << s << '\n';
        };

        if (a.tb.layer >= 0) {
          size_t k = 0;
          while (k < prog.calls.size() && prog.calls[k].layer != a.tb.layer) ++k;
          if (k == prog.calls.size())
            throw std::runtime_error("--layer " + std::to_string(a.tb.layer) + " : pas une conv");
          acc.load_input(x.data.data());
          for (int id = 0; id < a.tb.layer; ++id) {
            const LayerType t = m.layers[size_t(id)].type;
            const driver::View& v = prog.views[size_t(id)];
            if ((t == LayerType::Conv || t == LayerType::Maxpool) && plain(v))
              put(acc.arena(), v, tb::dump(a.tb, net, image, id));
          }
          log(prog.calls[k].layer, acc.run_layer(prog.calls[k]));
          check(k);
          std::printf("%s %s conv L%02d isolée : %s, %.3f ms\n", net.c_str(), image.c_str(),
                      a.tb.layer, nd_img ? "ÉCHEC" : "OK", 1e3 * t_acc);
          total_diff += nd_img;
          continue;
        }

        size_t k = 0;
        acc.run(x.data.data(), [&](const driver::ConvCall& c, double s) {
          log(c.layer, s);
          check(k++);
        });

        const auto t0 = std::chrono::steady_clock::now();
        const auto dets = sw::detect(m, prog, acc.arena());
        const double t_post = sw::seconds_since(t0);
        const std::string ref = tb::net_dir(a.tb, net) + "/dumps/" + image + "/detections.json";
        const bool same = same_detections(dets, read_detections(ref));
        if (!same) {
          std::printf("  détections DIFFÉRENTES de %s\n", ref.c_str());
          ++nd_img;
        }
        total_diff += nd_img;
        std::printf("%s %s : %s, %zu détections, accélérateur %.1f ms, post-traitement %.3f ms\n",
                    net.c_str(), image.c_str(), nd_img ? "ÉCHEC" : "OK", dets.size(),
                    1e3 * t_acc, 1e3 * t_post);
      }
    }
    std::printf("%d sorties de couche vérifiées, %zu écarts\n", checked, total_diff);
    return (total_diff == 0 && checked > 0) ? 0 : 1;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "erreur : %s\n", ex.what());
    return 1;
  }
}
