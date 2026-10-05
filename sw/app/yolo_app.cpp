// Démo de détection sur l'ARM (T7.3, T7.4) : image ou entrée int8 → accélérateur → boîtes.
//
//   yolo_app --model DIR (--input input.npy | --image photo.jpg) [--out detections.json]
//            [--draw boites.jpg] [--conf 0.25] [--repeat N] [--hw-post]
//            [--backend sim|uio] [--uio /dev/uioN] [--udmabuf udmabufN] [--poll]
//            [--uio-post /dev/uioN]
//
// --input : entrée déjà quantifiée (dumps/<id>/input.npy), sortie comparable bit à bit au
// dump. --image : JPEG/PNG/PPM, redimensionnement `stretch` Darknet (preprocess.hpp), boîtes
// normalisées dans l'image d'origine. Temps séparés : prétraitement, accélérateur,
// post-traitement (moyenne sur --repeat passes). --hw-post (T9.1) : décodage et NMS par le
// noyau `yolo_post` au lieu de l'ARM.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "accel_driver.hpp"
#include "golden/detections_io.hpp"
#include "golden/model.hpp"
#include "golden/npy.hpp"
#include "heads.hpp"
#include "options.hpp"
#include "preprocess.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

namespace {

const char* const VOC[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                           "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

int usage() {
  std::fprintf(stderr,
               "usage : yolo_app --model DIR (--input x.npy | --image img.jpg) [--out det.json]\n"
               "                 [--draw out.jpg] [--conf 0.25] [--repeat N] [--hw-post]\n"
               "                 [--backend sim|uio] [--uio /dev/uioN] [--udmabuf udmabufN] "
               "[--poll]\n");
  return 2;
}

// Cadre de 2 pixels autour de chaque boîte (cx, cy, w, h) normalisée.
void draw(uint8_t* rgb, int W, int H, const std::vector<postproc::Detection>& dets) {
  static const uint8_t color[][3] = {{230, 25, 75}, {60, 180, 75}, {0, 130, 200},
                                     {245, 130, 48}, {145, 30, 180}, {70, 240, 240}};
  for (const auto& d : dets) {
    const uint8_t* c = color[d.label % 6];
    const int x0 = std::max(0, int((d.box[0] - d.box[2] / 2) * W));
    const int x1 = std::min(W - 1, int((d.box[0] + d.box[2] / 2) * W));
    const int y0 = std::max(0, int((d.box[1] - d.box[3] / 2) * H));
    const int y1 = std::min(H - 1, int((d.box[1] + d.box[3] / 2) * H));
    auto set = [&](int x, int y) {
      if (x < 0 || y < 0 || x >= W || y >= H) return;
      std::copy(c, c + 3, rgb + (size_t(y) * W + x) * 3);
    };
    for (int t = 0; t < 2; ++t) {
      for (int x = x0; x <= x1; ++x) set(x, y0 + t), set(x, y1 - t);
      for (int y = y0; y <= y1; ++y) set(x0 + t, y), set(x1 - t, y);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  driver::DeviceOptions dopt;
  std::string model_dir, input, image, out, draw_path;
  double conf = 0.25;
  int repeat = 1;
  bool hw_post = false;
  for (int i = 1; i < argc; ++i) {
    if (sw::parse_device_option(argc, argv, i, dopt)) continue;
    const std::string k = argv[i];
    if (k == "--hw-post") {
      hw_post = true;
      continue;
    }
    if (i + 1 >= argc) return usage();
    const std::string v = argv[++i];
    if (k == "--model") model_dir = v;
    else if (k == "--input") input = v;
    else if (k == "--image") image = v;
    else if (k == "--out") out = v;
    else if (k == "--draw") draw_path = v;
    else if (k == "--conf") conf = std::atof(v.c_str());
    else if (k == "--repeat") repeat = std::max(1, std::atoi(v.c_str()));
    else return usage();
  }
  if (model_dir.empty() || input.empty() == image.empty()) return usage();

  try {
    const golden::Model m = golden::Model::load(model_dir);
    const auto dev = driver::make_device(dopt);
    driver::Accelerator acc(*dev, m);

    // Prétraitement
    auto t0 = std::chrono::steady_clock::now();
    std::vector<int8_t> x;
    int W = 0, H = 0;
    uint8_t* rgb = nullptr;
    if (!input.empty()) {
      x = golden::npy_load_int8(input).data;
    } else {
      int n = 0;
      rgb = stbi_load(image.c_str(), &W, &H, &n, 3);
      if (!rgb) throw std::runtime_error("lecture impossible : " + image);
      if (m.in_h != m.in_w || m.in_c != 3) throw std::runtime_error("entrée carrée RGB attendue");
      x = sw::preprocess_stretch(rgb, W, H, m.in_w, m.input_scale);
    }
    if (x.size() != size_t(m.in_c) * m.in_h * m.in_w)
      throw std::runtime_error("entrée de taille inattendue");
    const double t_pre = sw::seconds_since(t0);

    // Accélérateur, puis post-traitement sur l'ARM.
    double t_acc = 0.0, t_post = 0.0;
    std::vector<postproc::Detection> dets;
    for (int r = 0; r < repeat; ++r) {
      t0 = std::chrono::steady_clock::now();
      acc.run(x.data());
      t_acc += sw::seconds_since(t0);
      t0 = std::chrono::steady_clock::now();
      dets = hw_post ? sw::hw_detections(acc.run_post(conf, 0.45))
                     : sw::detect(m, acc.program(), acc.arena(), conf);
      t_post += sw::seconds_since(t0);
    }

    std::printf("%s (%s) : %zu détections\n", m.network.c_str(), dev->name(), dets.size());
    for (const auto& d : dets) {
      const char* name = m.classes == 20 ? VOC[d.label] : nullptr;
      std::printf("  %-12s %5.3f  cx %.3f cy %.3f w %.3f h %.3f\n",
                  name ? name : std::to_string(d.label).c_str(), d.score, d.box[0], d.box[1],
                  d.box[2], d.box[3]);
    }
    std::printf("temps : prétraitement %.2f ms, accélérateur %.2f ms, post-traitement %.3f ms "
                "(moyenne sur %d)\n",
                1e3 * t_pre, 1e3 * t_acc / repeat, 1e3 * t_post / repeat, repeat);
    if (!out.empty()) golden::write_detections(out, dets);
    if (!draw_path.empty()) {
      if (!rgb) throw std::runtime_error("--draw demande --image");
      draw(rgb, W, H, dets);
      if (!stbi_write_jpg(draw_path.c_str(), W, H, 3, rgb, 90))
        throw std::runtime_error("écriture impossible : " + draw_path);
    }
    if (rgb) stbi_image_free(rgb);
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "erreur : %s\n", ex.what());
    return 1;
  }
}
