// Testbench du post-traitement matériel (T9.1.2) : boîtes de `yolo_post` == `hwpp::run`
// (golden) sur les têtes des dumps et sur des têtes aléatoires ; cycles estimés en C-sim.
//
//   tb_post [--model DIR] [--net NAME]… [--image ID]… [--random N]
#include <cstdio>
#include <random>

#include "golden/hw_heads.hpp"
#include "golden/model.hpp"
#include "post_table.hpp"
#include "postproc.hpp"
#include "tb_common.hpp"

using namespace golden;

static bool same(const std::vector<hwpp::Box>& a, const std::vector<hwpp::Box>& b) {
  if (a.size() != b.size()) return false;
  for (size_t k = 0; k < a.size(); ++k)
    if (a[k].x1 != b[k].x1 || a[k].y1 != b[k].y1 || a[k].x2 != b[k].x2 || a[k].y2 != b[k].y2 ||
        a[k].score != b[k].score || a[k].cls != b[k].cls)
      return false;
  return true;
}

// Une image : têtes bout à bout dans une arène, appel du noyau, comparaison au golden.
static bool check(const Model& m, const std::vector<std::vector<int8_t>>& heads, double conf,
                  const char* what, bool verbose) {
  std::vector<int8_t> arena;
  std::vector<int64_t> off;
  std::vector<hwpp::HeadData> ref;
  for (const auto& h : heads) {
    off.push_back(int64_t(arena.size()));
    arena.insert(arena.end(), h.begin(), h.end());
  }
  const std::vector<int> ids = m.heads();
  for (size_t k = 0; k < heads.size(); ++k)
    ref.push_back(make_hw_head(m, m.layers[size_t(ids[k])], arena.data() + off[k], conf));
  int ov_ref = 0;
  const auto want = hwpp::run(ref, make_hw_params(conf, 0.45), &ov_ref);

  const int32_t base = 16;  // table à un indice non nul : les offsets sont bien relatifs
  driver::PostTable t = driver::post_table(m, off, conf, 0.45, base, 3);
  std::vector<int32_t> tab(size_t(base), -1);
  tab.insert(tab.end(), t.words.begin(), t.words.end());
  std::vector<int32_t> res(size_t(3 + 2 + accel::POST_BOX_WORDS * accel::POST_CAP), -7);
  yolo_post(arena.data(), tab.data(), res.data(), t.desc);
  int ov = 0;
  const auto got = driver::post_boxes(res.data() + 3, &ov);
  const bool ok = same(got, want) && ov == ov_ref;
  if (verbose || !ok) {
    const accel::PostCycles& c = accel::post_cycles;
    std::printf("  %s conf %.3f : %zu boîtes, %d candidates, %d survivantes, débordement %d, "
                "≈ %llu cycles (scan %llu, décodage %llu, NMS %llu) %s\n",
                what, conf, got.size(), c.candidates, c.survivors, ov,
                (unsigned long long)c.total, (unsigned long long)c.scan,
                (unsigned long long)c.decode, (unsigned long long)c.nms, ok ? "OK" : "ÉCHEC");
  }
  return ok;
}

int main(int argc, char** argv) {
  const tb::Args a = tb::parse(argc, argv);
  int checked = 0, failed = 0;
  std::mt19937 rng(7);
  for (const std::string& net : a.nets) {
    if (!tb::have_model(a, net)) {
      std::printf("%s : export absent (make export), sauté\n", net.c_str());
      continue;
    }
    const Model m = Model::load(tb::net_dir(a, net));
    std::printf("%s\n", net.c_str());
    for (const std::string& image : a.images)
      for (double conf : {0.25, 0.005}) {
        std::vector<std::vector<int8_t>> heads;
        for (int id : m.heads()) heads.push_back(tb::dump(a, net, image, id).data);
        failed += !check(m, heads, conf, image.c_str(), true);
        ++checked;
      }
    // Pire cas de 2024-zhang : toutes les cellules passent le seuil d'objectness.
    {
      std::vector<std::vector<int8_t>> heads;
      for (int id : m.heads()) {
        const Layer& l = m.layers[size_t(id)];
        std::vector<int8_t> h(size_t(l.out_c) * l.out_h * l.out_w);
        std::uniform_int_distribution<int> u(-128, 127);
        for (auto& v : h) v = int8_t(u(rng));
        const int A = l.out_c / (5 + m.classes), plane = l.out_h * l.out_w;
        for (int an = 0; an < A; ++an)
          for (int c = 0; c < plane; ++c) h[size_t((an * (5 + m.classes) + 4) * plane + c)] = 127;
        heads.push_back(h);
      }
      failed += !check(m, heads, 0.25, "toutes cellules", true);
      ++checked;
    }
    // Têtes aléatoires ; --random N pour une campagne longue (T12.7, ex. 100000).
    const int n_rand = a.random >= 0 ? a.random : net == "tiny-yolov2-voc" ? 800 : 200;
    for (int r = 0; r < n_rand; ++r) {
      std::vector<std::vector<int8_t>> heads;
      std::uniform_int_distribution<int> u(-128, 127), bias(-60, 40);
      const int obj_bias = bias(rng);
      for (int id : m.heads()) {
        const Layer& l = m.layers[size_t(id)];
        std::vector<int8_t> h(size_t(l.out_c) * l.out_h * l.out_w);
        for (auto& v : h) v = int8_t(u(rng));
        const int A = l.out_c / (5 + m.classes), plane = l.out_h * l.out_w;
        for (int an = 0; an < A; ++an)
          for (int c = 0; c < plane; ++c) {
            int8_t& o = h[size_t((an * (5 + m.classes) + 4) * plane + c)];
            o = int8_t(std::max(-128, std::min(127, int(o) / 2 + obj_bias)));
          }
        heads.push_back(h);
      }
      const double confs[] = {0.005, 0.1, 0.25, 0.5};
      failed += !check(m, heads, confs[r % 4], "aléatoire", false);
      ++checked;
    }
  }
  std::printf("%d cas vérifiés, %d échecs\n", checked, failed);
  return (failed == 0 && checked > 0) ? 0 : 1;
}
