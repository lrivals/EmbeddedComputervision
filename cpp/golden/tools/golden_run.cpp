// Golden model en ligne de commande (T5.1, T5.4).
//
//   golden_run run <model_dir> <input.npy> <out_dir>   → Lxx.npy + detections.json
//   golden_run inspect <model_dir>                     → résumé du modèle relu (T5.1)
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

#include "golden/engine.hpp"
#include "golden/model.hpp"
#include "golden/npy.hpp"

using namespace golden;

static int usage() {
  std::cerr << "usage : golden_run run <model_dir> <input.npy> <out_dir>\n"
               "        golden_run inspect <model_dir>\n";
  return 2;
}

template <class T>
static void stats(const T* p, int64_t n, long long& sum) {
  sum = 0;
  for (int64_t i = 0; i < n; ++i) sum += p[i];
}

static int inspect(const std::string& dir) {
  const Model m = Model::load(dir);
  std::printf("network %s classes %d input %d %d %d scale %.17g\n", m.network.c_str(), m.classes,
              m.in_c, m.in_h, m.in_w, m.input_scale);
  for (const auto& a : m.anchors) std::printf("anchor %.17g %.17g\n", a.first, a.second);
  for (const auto& b : m.buffers) std::printf("buffer %s %lld\n", b.first.c_str(), (long long)b.second);
  for (const Layer& l : m.layers) {
    std::printf("layer %d %s out %d %d %d", l.id, layer_type_name(l.type), l.out_c, l.out_h,
                l.out_w);
    if (l.type == LayerType::Conv) {
      const int64_t nw = int64_t(l.cout) * l.cin * l.k * l.k;
      long long sw, sb, sm;
      stats(m.conv_weights(l), nw, sw);
      stats(m.conv_bias(l), l.cout, sb);
      stats(m.conv_m0(l), l.cout, sm);
      std::printf(" k %d pad %d cin %d cout %d leaky %d pool_k %d pool_s %d shift %d in_scale %.17g "
                  "out_scale %.17g w_sum %lld w_first %d w_last %d b_sum %lld b_first %d "
                  "m0_sum %lld m0_first %d",
                  l.k, l.pad, l.cin, l.cout, int(l.leaky), l.pool_k, l.pool_s, l.shift,
                  l.in_scale, l.out_scale, sw, int(m.conv_weights(l)[0]),
                  int(m.conv_weights(l)[nw - 1]), sb, m.conv_bias(l)[0], sm, m.conv_m0(l)[0]);
    } else if (l.is_head()) {
      long long s0;
      stats(m.head_lut(l, 0), 3 * 256, s0);
      std::printf(" scale %.17g exp_frac %d lut_sum %lld", l.scale, l.exp_frac, s0);
    } else if (l.type != LayerType::Maxpool) {
      std::printf(" scale %.17g", l.scale);
    }
    std::printf("\n");
  }
  return 0;
}

static void write_detections(const std::string& path, const std::vector<postproc::Detection>& d) {
  FILE* f = std::fopen(path.c_str(), "w");
  if (!f) throw std::runtime_error("impossible d'écrire " + path);
  std::fprintf(f, "{\"boxes\": [");
  for (size_t k = 0; k < d.size(); ++k)
    std::fprintf(f, "%s[%.17g, %.17g, %.17g, %.17g]", k ? ", " : "", d[k].box[0], d[k].box[1],
                 d[k].box[2], d[k].box[3]);
  std::fprintf(f, "], \"scores\": [");
  for (size_t k = 0; k < d.size(); ++k) std::fprintf(f, "%s%.17g", k ? ", " : "", d[k].score);
  std::fprintf(f, "], \"labels\": [");
  for (size_t k = 0; k < d.size(); ++k) std::fprintf(f, "%s%d", k ? ", " : "", d[k].label);
  std::fprintf(f, "]}\n");
  std::fclose(f);
}

static int run(const std::string& dir, const std::string& input, const std::string& out) {
  const Model m = Model::load(dir);
  const NpyInt8 x = npy_load_int8(input);
  if (x.numel() != size_t(m.in_c) * m.in_h * m.in_w)
    throw std::runtime_error("entrée de taille inattendue");
  Engine e(m);
  e.run(x.data.data(), true);
  std::filesystem::create_directories(out);
  for (size_t i = 0; i < e.recorded().size(); ++i) {
    const Tensor& t = e.recorded()[i];
    char name[32];
    std::snprintf(name, sizeof name, "L%02zu.npy", i);
    npy_save_int8(out + "/" + name, {1, t.c, t.h, t.w}, t.data.data());
  }
  const auto dets = postproc::postprocess(make_heads(e));
  write_detections(out + "/detections.json", dets);
  std::printf("%s : %zu couches, %zu détections → %s\n", m.network.c_str(), e.recorded().size(),
              dets.size(), out.c_str());
  return 0;
}

int main(int argc, char** argv) {
  try {
    if (argc == 5 && std::strcmp(argv[1], "run") == 0) return run(argv[2], argv[3], argv[4]);
    if (argc == 3 && std::strcmp(argv[1], "inspect") == 0) return inspect(argv[2]);
    return usage();
  } catch (const std::exception& ex) {
    std::cerr << "erreur : " << ex.what() << "\n";
    return 1;
  }
}
