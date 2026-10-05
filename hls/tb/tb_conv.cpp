// Testbench par couche (T6.1, T6.3) : chaque conv des deux réseaux, seule, sur l'entrée
// dumpée par Python. Sortie du noyau == golden::conv_layer (mêmes tuiles) == dump Python,
// pour la carte poolée et la carte avant pooling (maxpool 2×2 stride 2 et 1).
//
//   tb_conv [--model DIR] [--net NAME]… [--image ID]… [--layer N] [--csv cycles.csv]
//
// Code de sortie 0 si aucun écart (convention C-sim Vitis). Imprime aussi l'estimation de
// cycles par couche (accel::sim_cycles) pour la première image.
#include <cstdio>
#include <cstring>
#include <fstream>

#include "accel.hpp"
#include "golden/conv.hpp"
#include "golden/engine.hpp"
#include "golden/model.hpp"
#include "program.hpp"
#include "tb_common.hpp"

#ifndef ACC_FREQ_MHZ
#define ACC_FREQ_MHZ 200
#endif

using namespace golden;
using Tl = Tiles<accel::TM, accel::TN, accel::TR, accel::TC>;

int main(int argc, char** argv) {
  const tb::Args a = tb::parse(argc, argv);
  std::printf("tuiles Tm %d Tn %d Tr %d Tc %d\n", accel::TM, accel::TN, accel::TR, accel::TC);
  std::ofstream csv;
  if (!a.csv.empty()) {
    csv.open(a.csv);
    csv << "net,layer,k,cin,cout,h,w,pool_s,macs,tm,tn,tr,tc,fold,width,trim,rq,load_in,load_w,"
           "compute,store,sequential,overlapped\n";
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
    const int64_t m0_base = int64_t(m.bias.size());
    std::vector<driver::Word> wts((prog.weights.size() + accel::WORD - 1) / accel::WORD, 0);
    std::memcpy(wts.data(), prog.weights.data(), prog.weights.size());
    uint64_t net_ovl = 0, net_comp = 0;

    for (size_t ii = 0; ii < a.images.size(); ++ii) {
      const std::string& image = a.images[ii];
      for (const Layer& l : m.layers) {
        if (l.type != LayerType::Conv || (a.layer >= 0 && l.id != a.layer)) continue;
        const NpyInt8 x = tb::dump(a, net, image, l.id - 1);
        const int c = int(x.shape[1]), h = int(x.shape[2]), w = int(x.shape[3]);
        const ConvParams p = conv_params(l, h, w);
        const int64_t n_in = int64_t(c) * h * w;
        const int64_t n_out = int64_t(p.cout) * p.pool_h() * p.pool_w();
        const int64_t n_pre = p.pooled() ? int64_t(p.cout) * p.out_h() * p.out_w() : 0;

        // Arène : entrée | sortie | carte avant pooling, bout à bout (sans alignement : le
        // chargeur et les écritures en mots gèrent tout décalage), puis la marge.
        const int64_t n_all = n_in + n_out + n_pre + driver::ARENA_SLACK;
        std::vector<driver::Word> arena(size_t(n_all + accel::WORD - 1) / accel::WORD, 0);
        int8_t* bytes = reinterpret_cast<int8_t*>(arena.data());
        std::copy(x.data.begin(), x.data.end(), bytes);
        const driver::View in = driver::single_view(0, c, h, w);
        const accel::LayerDesc d =
            driver::conv_desc(l, in, n_in, p.pooled() ? n_in + n_out : -1, l.b_offset / 4,
                              m0_base + l.m0_offset / 4, prog.w_off.at(l.id));
        yolo_conv(arena.data(), arena.data(), wts.data(), prog.params.data(), d, nullptr, 0);
        const int8_t* k_out = bytes + n_in;
        const int8_t* k_pre = bytes + n_in + n_out;

        std::vector<int8_t> g_out(static_cast<size_t>(n_out)), g_pre(static_cast<size_t>(n_pre));
        conv_layer<Tl>(p, golden::single_view(x.data.data(), c, h, w), m.conv_weights(l),
                       m.conv_bias(l), m.conv_m0(l), g_out.data(), nullptr,
                       p.pooled() ? g_pre.data() : nullptr);

        std::printf("%s %s L%02d : ", net.c_str(), image.c_str(), l.id);
        size_t nd = tb::compare("golden", k_out, g_out.data(), size_t(n_out));
        const int out_id = p.pooled() ? l.pool_layer : l.id;
        const NpyInt8 want = tb::dump(a, net, image, out_id);
        nd += tb::compare("dump", k_out, want.data.data(), size_t(n_out));
        if (p.pooled()) {
          nd += tb::compare("golden prépool", k_pre, g_pre.data(), size_t(n_pre));
          const NpyInt8 want_pre = tb::dump(a, net, image, l.id);
          nd += tb::compare("dump prépool", k_pre, want_pre.data.data(), size_t(n_pre));
        }
        total_diff += nd;
        ++checked;

        const accel::SimCycles& s = accel::sim_cycles;
        const double macs = double(p.cout) * p.cin * p.out_h() * p.out_w() * p.k * p.k;
        std::printf("%s  cycles ≈ %llu (ping-pong) / %llu (séquentiel), calcul %llu, %.2f ms\n",
                    nd ? "ÉCHEC" : "OK", (unsigned long long)s.overlapped,
                    (unsigned long long)s.sequential, (unsigned long long)s.compute,
                    s.overlapped / (ACC_FREQ_MHZ * 1e3));
        if (ii == 0) {
          net_ovl += s.overlapped;
          net_comp += s.compute;
          if (csv.is_open())
            csv << net << ',' << l.id << ',' << p.k << ',' << p.cin << ',' << p.cout << ','
                << h << ',' << w << ',' << p.pool_s << ',' << (long long)macs << ','
                << accel::TM << ',' << accel::TN << ',' << d.tr << ',' << d.tc << ','
                << d.fold << ',' << accel::WORD << ',' << int(accel::TRIM) << ','
                << accel::RQ << ',' << s.load_in << ',' << s.load_w << ',' << s.compute << ',' << s.store << ','
                << s.sequential << ',' << s.overlapped << '\n';
        }
      }
    }
    if (a.layer < 0)
      std::printf("%s : ≈ %llu cycles (ping-pong), %.1f ms à %d MHz ; calcul seul %.1f ms\n",
                  net.c_str(), (unsigned long long)net_ovl, net_ovl / (ACC_FREQ_MHZ * 1e3),
                  ACC_FREQ_MHZ, net_comp / (ACC_FREQ_MHZ * 1e3));
  }
  std::printf("%d couches vérifiées, %zu écarts\n", checked, total_diff);
  return (total_diff == 0 && checked > 0) ? 0 : 1;
}
