// Faults in the on-chip buffers of `yolo_conv` and their detection by the ABFT lane
// (T25.3, T25.7). One conv at a time, on the input dumped by Python (as tb_conv); for each
// flip, the kernel runs once with the C-sim fault hook armed (hls/kernels/fault_hook.hpp):
// output compared with the fault-free run, ABFT register read.
//
//   tb_fault                                   ctest: Tiny-YOLOv2, a few layers and flips;
//                                              fault-free → 0 ABFT error; an accumulator flip
//                                              → detected; a checksum-lane flip → detected,
//                                              outputs unchanged
//   tb_fault --net N --flips F [--layer L]… [--csv F] [--dump DIR] [--seed S]
//
// --csv: one line per flip (buffer, indices, tile geometry, bit, bytes changed, ABFT pixels);
// --dump: also the faulty output of each flip (DIR/<net>/fNNNNN.npy) for the Python reference
// of python/tests/test_faults.py (buffer address → tensor index, T25.7).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "accel.hpp"
#include "golden/engine.hpp"
#include "output_stage.hpp"
#include "golden/model.hpp"
#include "golden/npy.hpp"
#include "program.hpp"
#include "tb_common.hpp"

using namespace golden;
using accel::HlsFault;

namespace {

const char* const BUF_NAME[] = {"in", "w", "out", "q", "cs"};

struct Opt {
  tb::Args a;
  std::vector<int> layers;
  int flips = -1;  // -1: ctest mode
  unsigned seed = 25;
  std::string csv, dump;
};

Opt parse(int argc, char** argv) {
  Opt o;
  std::vector<char*> rest = {argv[0]};
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    if (i + 1 < argc && k == "--flips") o.flips = std::atoi(argv[++i]);
    else if (i + 1 < argc && k == "--layer") o.layers.push_back(std::atoi(argv[++i]));
    else if (i + 1 < argc && k == "--seed") o.seed = unsigned(std::atoi(argv[++i]));
    else if (i + 1 < argc && k == "--csv") o.csv = argv[++i];
    else if (i + 1 < argc && k == "--dump") o.dump = argv[++i];
    else rest.push_back(argv[i]);
  }
  o.a = tb::parse(int(rest.size()), rest.data());
  return o;
}

// Geometry of output tile `id` of a conv (same formulas as accel::conv_layer).
struct Geo {
  int row, col, prow0, pcol0, to, tr_n, tc_n, np_r, np_c, tm_n, im, nti;
};

struct Layout {
  int n_r, n_c, n_m, n_tiles;
  Geo tile(const accel::LayerDesc& d, int id) const {
    const int pk = d.pk(), ps = d.ps(), lo = d.pool_lo();
    const int R = d.out_h(), C = d.out_w(), Rp = d.pool_h(), Cp = d.pool_w();
    const int Pr = (d.tr - pk) / ps + 1, Pc = (d.tc - pk) / ps + 1;
    Geo g{};
    g.im = id % n_m;
    const int ic = (id / n_m) % n_c, ir = id / (n_m * n_c);
    g.prow0 = ir * Pr;
    g.pcol0 = ic * Pc;
    g.row = accel::imax(g.prow0 * ps - lo, 0);
    g.col = accel::imax(g.pcol0 * ps - lo, 0);
    g.to = g.im * accel::TM;
    g.tr_n = accel::imin((g.prow0 + Pr - 1) * ps - lo + pk - 1, R - 1) - g.row + 1;
    g.tc_n = accel::imin((g.pcol0 + Pc - 1) * ps - lo + pk - 1, C - 1) - g.col + 1;
    g.np_r = accel::imin(Pr, Rp - g.prow0);
    g.np_c = accel::imin(Pc, Cp - g.pcol0);
    g.tm_n = accel::imin(accel::TM, d.cout - g.to);
    g.nti = accel::n_ti(d);
    return g;
  }
};

Layout layout(const accel::LayerDesc& d) {
  const int pk = d.pk(), ps = d.ps();
  const int Pr = (d.tr - pk) / ps + 1, Pc = (d.tc - pk) / ps + 1;
  Layout l;
  l.n_r = (d.pool_h() + Pr - 1) / Pr;
  l.n_c = (d.pool_w() + Pc - 1) / Pc;
  l.n_m = (d.cout + accel::TM - 1) / accel::TM;
  l.n_tiles = l.n_r * l.n_c * l.n_m;
  return l;
}

// A random flip of buffer `buf` at a location the kernel uses for this tile.
HlsFault draw(std::mt19937& rng, int buf, const accel::LayerDesc& d, const Layout& L, Geo& g) {
  auto u = [&](int n) { return n > 0 ? int(rng() % unsigned(n)) : 0; };
  HlsFault f;
  f.buf = buf;
  f.tile = buf == accel::FB_CS ? u(L.n_r * L.n_c) * L.n_m : u(L.n_tiles);
  g = L.tile(d, f.tile);
  f.ti = u(g.nti);
  const int n = accel::ti_lanes(d, f.ti), kh = accel::kern_h(d);
  const int rows = d.fold ? g.tr_n : g.tr_n + d.k - 1, cols = g.tc_n + d.k - 1;
  switch (buf) {
    case accel::FB_IN:  // lane, row, column of the loaded window
      f.a = u(n), f.b = u(rows), f.c = u(cols), f.bit = u(8);
      break;
    case accel::FB_W:
      f.a = u(g.tm_n), f.b = u(n), f.c = u(kh), f.d = u(d.k), f.bit = u(8);
      break;
    case accel::FB_OUT:
      f.a = u(g.tm_n), f.b = u(g.tr_n), f.c = u(g.tc_n), f.bit = u(32);
      break;
    case accel::FB_Q:
      f.a = u(g.tm_n), f.b = u(g.tr_n), f.c = u(g.tc_n), f.bit = u(8);
      break;
    default:  // FB_CS
      f.b = u(g.tr_n), f.c = u(g.tc_n), f.bit = u(32);
  }
  return f;
}

struct Run {
  std::vector<int8_t> out;
  int32_t abft = 0;
};

}  // namespace

int main(int argc, char** argv) {
  Opt o = parse(argc, argv);
  const bool ctest = o.flips < 0;
  if (ctest) {
    o.a.nets = {"tiny-yolov2-voc"};
    if (o.layers.empty()) o.layers = {4, 8};
    o.flips = 2;
  }
  std::mt19937 rng(o.seed);
  std::ofstream csv;
  if (!o.csv.empty()) {
    csv.open(o.csv);
    csv << "net,image,layer,flip,buf,tile,ti,a,b,c,d,bit,fired,row,col,prow0,pcol0,to,tr_n,"
           "tc_n,np_r,np_c,tm_n,changed,abft\n";
  }
  int fails = 0, flips = 0;
  const std::string image = o.a.images.front();
  for (const std::string& net : o.a.nets) {
    if (!tb::have_model(o.a, net)) {
      std::printf("%s : export absent, sauté\n", net.c_str());
      continue;
    }
    const Model m = Model::load(tb::net_dir(o.a, net));
    const driver::Program prog = driver::build(m);
    std::vector<driver::Word> wts((prog.weights.size() + accel::WORD - 1) / accel::WORD, 0);
    std::memcpy(wts.data(), prog.weights.data(), prog.weights.size());
    if (!o.dump.empty()) std::filesystem::create_directories(o.dump + "/" + net);
    int flip_id = 0;
    for (const Layer& l : m.layers) {
      if (l.type != LayerType::Conv) continue;
      if (!o.layers.empty() && std::find(o.layers.begin(), o.layers.end(), l.id) == o.layers.end())
        continue;
      const NpyInt8 x = tb::dump(o.a, net, image, l.id - 1);
      const int c = int(x.shape[1]), h = int(x.shape[2]), w = int(x.shape[3]);
      const ConvParams p = conv_params(l, h, w);
      const int64_t n_in = int64_t(c) * h * w;
      const int64_t n_out = int64_t(p.cout) * p.pool_h() * p.pool_w();
      const driver::View in = driver::single_view(0, c, h, w);
      const accel::LayerDesc d = driver::conv_desc(l, in, n_in, -1, l.b_offset / 4,
                                                   prog.m0_index.at(l.id), prog.w_off.at(l.id));
      auto run = [&]() {
        std::vector<driver::Word> arena(size_t(n_in + n_out + driver::ARENA_SLACK + accel::WORD) /
                                            accel::WORD,
                                        0);
        int8_t* bytes = reinterpret_cast<int8_t*>(arena.data());
        std::copy(x.data.begin(), x.data.end(), bytes);
        Run r;
        yolo_conv(arena.data(), arena.data(), wts.data(), prog.params.data(), d, nullptr, 0,
                  &r.abft);
        r.out.assign(bytes + n_in, bytes + n_in + n_out);
        return r;
      };
      accel::hls_fault = HlsFault{};
      const Run clean = run();
      if (clean.abft != 0) {
        std::printf("%s L%02d : %d ABFT errors without fault\n", net.c_str(), l.id, clean.abft);
        ++fails;
      }
      const Layout L = layout(d);
      int detected[5] = {}, changed_n[5] = {}, total[5] = {};
      for (int buf = accel::FB_IN; buf <= accel::FB_CS; ++buf)
        for (int k = 0; k < o.flips; ++k) {
          Geo g;
          accel::hls_fault = draw(rng, buf, d, L, g);
          const HlsFault f = accel::hls_fault;
          const Run r = run();
          const bool fired = accel::hls_fault.fired != 0;
          int64_t changed = 0;
          for (int64_t i = 0; i < n_out; ++i) changed += r.out[size_t(i)] != clean.out[size_t(i)];
          ++total[buf];
          detected[buf] += r.abft > 0;
          changed_n[buf] += changed > 0;
          ++flips;
          if (csv.is_open())
            csv << net << ',' << image << ',' << l.id << ',' << flip_id << ',' << BUF_NAME[buf]
                << ',' << f.tile << ',' << f.ti << ',' << f.a << ',' << f.b << ',' << f.c << ','
                << f.d << ',' << f.bit << ',' << int(fired) << ',' << g.row << ',' << g.col << ','
                << g.prow0 << ',' << g.pcol0 << ',' << g.to << ',' << g.tr_n << ',' << g.tc_n
                << ',' << g.np_r << ',' << g.np_c << ',' << g.tm_n << ',' << changed << ','
                << r.abft << '\n';
          if (!o.dump.empty()) {
            char name[32];
            std::snprintf(name, sizeof name, "/f%05d.npy", flip_id);
            npy_save_int8(o.dump + "/" + net + name, {p.cout, p.pool_h(), p.pool_w()},
                          r.out.data());
          }
          ++flip_id;
          if (!fired) {
            std::printf("%s L%02d %s : fault not fired\n", net.c_str(), l.id, BUF_NAME[buf]);
            ++fails;
          }
          // Checks of the ctest mode: an accumulator flip changes the checked sum (detected);
          // a flip of the checksum lane is a false alarm that leaves the outputs alone.
          if (buf == accel::FB_OUT && r.abft == 0) {
            std::printf("%s L%02d out flip not detected\n", net.c_str(), l.id);
            ++fails;
          }
          if (buf == accel::FB_CS && (r.abft == 0 || changed)) {
            std::printf("%s L%02d cs flip: abft %d, %lld bytes changed\n", net.c_str(), l.id,
                        r.abft, (long long)changed);
            ++fails;
          }
        }
      std::printf("%s L%02d :", net.c_str(), l.id);
      for (int buf = 0; buf < 5; ++buf)
        std::printf(" %s %d/%d changed %d/%d detected", BUF_NAME[buf], changed_n[buf], total[buf],
                    detected[buf], total[buf]);
      std::printf("\n");
    }
  }
  accel::hls_fault = HlsFault{};
  std::printf("%d flips, %d failures\n", flips, fails);
  return fails == 0 && flips > 0 ? 0 : 1;
}
