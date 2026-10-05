// Mesures sur l'ARM (T8.1, T8.2) : passe complète sur une suite d'images, temps par étage et
// par image, détections au seuil de la mAP, puissance de la carte.
//
//   yolo_bench --model DIR (--inputs inputs.bin --ids ids.txt | --images liste.txt)
//              [--start 0] [--count N] [--warmup 5] [--conf 0.005] [--iou 0.45]
//              [--times temps.csv] [--dets det.jsonl] [--hw-post]
//              [--power /sys/class/hwmon/hwmonN/power1_input] [--idle-s 5]
//              [--backend sim|uio] [--uio /dev/uioN] [--udmabuf udmabufN] [--poll]
//
// --inputs : entrées int8 (3, S, S) concaténées, produites par tools/make_inputs.py avec le
// prétraitement du modèle entier Python : sortie comparable bit à bit (mAP au stade FPGA).
// Un `--inputs` en .npy (une ou quelques entrées, ex. dumps/<id>/input.npy) est accepté.
// --images : un chemin JPEG/PNG par ligne, prétraitement `stretch` sur l'ARM (preprocess.hpp)
// chronométré ; c'est le mode des mesures de latence (results/protocole.md).
//
// Étages (ms, par image) : pre = lecture de l'entrée (et décodage + redimensionnement +
// quantification en --images) ; load = mise à zéro de l'arène et copie de l'entrée en DDR ;
// acc = somme des convs (registres + calcul + attente) ; post = décodage + NMS (sur l'ARM, ou
// par le noyau `yolo_post` avec --hw-post, T9.1).
// --power : µW lus dans hwmon (INA260 du SOM KV260) toutes les 10 ms, au repos pendant
// --idle-s secondes puis pendant la boucle mesurée.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "accel_driver.hpp"
#include "golden/detections_io.hpp"
#include "golden/model.hpp"
#include "golden/npy.hpp"
#include "heads.hpp"
#include "options.hpp"
#include "preprocess.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace {

int usage() {
  std::fprintf(stderr,
               "usage : yolo_bench --model DIR (--inputs x.bin --ids ids.txt | --images l.txt)\n"
               "                   [--start 0] [--count N] [--warmup 5] [--conf 0.005]\n"
               "                   [--iou 0.45] [--times t.csv] [--dets d.jsonl] [--hw-post]\n"
               "                   [--power .../power1_input] [--idle-s 5]\n"
               "                   [--backend sim|uio] [--uio /dev/uioN] [--udmabuf udmabufN] "
               "[--poll]\n");
  return 2;
}

std::vector<std::string> read_lines(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("lecture impossible : " + path);
  std::vector<std::string> out;
  for (std::string l; std::getline(f, l);)
    if (!l.empty()) out.push_back(l);
  return out;
}

std::string stem(const std::string& path) {
  const size_t a = path.find_last_of('/');
  std::string s = a == std::string::npos ? path : path.substr(a + 1);
  return s.substr(0, s.find_last_of('.'));
}

// Puissance moyenne (W) lue dans un fichier hwmon en µW, échantillonnée en tâche de fond.
class PowerSampler {
 public:
  explicit PowerSampler(std::string path) : path_(std::move(path)) {}
  ~PowerSampler() { stop(); }
  void start() {
    sum_ = 0.0;
    n_ = 0;
    run_ = true;
    th_ = std::thread([this] {
      while (run_) {
        std::ifstream f(path_);
        double uw = 0.0;
        if (f >> uw) sum_ += uw * 1e-6, ++n_;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    });
  }
  double stop() {
    if (th_.joinable()) {
      run_ = false;
      th_.join();
    }
    return n_ ? sum_ / double(n_) : 0.0;
  }
  long samples() const { return n_; }

 private:
  std::string path_;
  std::thread th_;
  std::atomic<bool> run_{false};
  double sum_ = 0.0;
  long n_ = 0;
};

double mean(const std::vector<double>& v) {
  double s = 0.0;
  for (double x : v) s += x;
  return v.empty() ? 0.0 : s / double(v.size());
}

// Quantile empirique « nearest rank » (p99 sur 1 000 images = 990e valeur triée).
double quantile(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  size_t k = size_t(std::max(1.0, std::ceil(q * double(v.size()))));
  return v[std::min(k, v.size()) - 1];
}

}  // namespace

int main(int argc, char** argv) {
  driver::DeviceOptions dopt;
  std::string model_dir, inputs, ids_path, images, times_path, dets_path, power_path;
  double conf = 0.005, iou = 0.45, idle_s = 5.0;
  long start = 0, count = -1;
  int warmup = 5;
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
    else if (k == "--inputs") inputs = v;
    else if (k == "--ids") ids_path = v;
    else if (k == "--images") images = v;
    else if (k == "--start") start = std::atol(v.c_str());
    else if (k == "--count") count = std::atol(v.c_str());
    else if (k == "--warmup") warmup = std::max(0, std::atoi(v.c_str()));
    else if (k == "--conf") conf = std::atof(v.c_str());
    else if (k == "--iou") iou = std::atof(v.c_str());
    else if (k == "--times") times_path = v;
    else if (k == "--dets") dets_path = v;
    else if (k == "--power") power_path = v;
    else if (k == "--idle-s") idle_s = std::atof(v.c_str());
    else return usage();
  }
  if (model_dir.empty() || inputs.empty() == images.empty()) return usage();
  if (!inputs.empty() && ids_path.empty()) return usage();

  try {
    const golden::Model m = golden::Model::load(model_dir);
    const auto dev = driver::make_device(dopt);
    driver::Accelerator acc(*dev, m);
    const size_t in_bytes = size_t(m.in_c) * m.in_h * m.in_w;
    if (!images.empty() && (m.in_h != m.in_w || m.in_c != 3))
      throw std::runtime_error("entrée carrée RGB attendue");

    const std::vector<std::string> list = read_lines(inputs.empty() ? images : ids_path);
    const long n_all = long(list.size());
    if (start < 0 || start > n_all) throw std::runtime_error("--start hors de la liste");
    const long end = count < 0 ? n_all : std::min(n_all, start + count);

    // .npy : petites suites (tests, dumps), chargées en mémoire ; .bin : lues image par image.
    const bool npy = inputs.size() > 4 && inputs.compare(inputs.size() - 4, 4, ".npy") == 0;
    std::vector<int8_t> npy_all;
    std::ifstream bin;
    if (npy) {
      npy_all = golden::npy_load_int8(inputs).data;
      if (npy_all.size() != size_t(n_all) * in_bytes)
        throw std::runtime_error(inputs + " : taille différente de " + std::to_string(n_all) +
                                 " entrées");
    } else if (!inputs.empty()) {
      bin.open(inputs, std::ios::binary | std::ios::ate);
      if (!bin) throw std::runtime_error("lecture impossible : " + inputs);
      if (size_t(bin.tellg()) != size_t(n_all) * in_bytes)
        throw std::runtime_error(inputs + " : taille différente de " + std::to_string(n_all) +
                                 " entrées de " + std::to_string(in_bytes) + " octets");
    }

    // Lecture (et prétraitement) de l'image k → entrée int8 ; rend l'identifiant.
    std::vector<int8_t> x(in_bytes);
    auto fetch = [&](long k) -> std::string {
      if (npy) {
        std::copy_n(npy_all.begin() + std::ptrdiff_t(size_t(k) * in_bytes), in_bytes, x.begin());
        return list[size_t(k)];
      }
      if (!inputs.empty()) {
        bin.seekg(std::streamoff(k) * std::streamoff(in_bytes));
        if (!bin.read(reinterpret_cast<char*>(x.data()), std::streamsize(in_bytes)))
          throw std::runtime_error(inputs + " : lecture de l'entrée " + std::to_string(k));
        return list[size_t(k)];
      }
      int W = 0, H = 0, n = 0;
      uint8_t* rgb = stbi_load(list[size_t(k)].c_str(), &W, &H, &n, 3);
      if (!rgb) throw std::runtime_error("lecture impossible : " + list[size_t(k)]);
      x = sw::preprocess_stretch(rgb, W, H, m.in_w, m.input_scale);
      stbi_image_free(rgb);
      return stem(list[size_t(k)]);
    };
    auto forward = [&]() {
      acc.load_input(x.data());
      double s = 0.0;
      for (const driver::ConvCall& c : acc.program().calls) s += acc.run_layer(c);
      return s;
    };

    std::FILE* times = nullptr;
    std::FILE* dets = nullptr;
    if (!times_path.empty()) {
      times = std::fopen(times_path.c_str(), "w");
      if (!times) throw std::runtime_error("impossible d'écrire " + times_path);
      std::fprintf(times, "id,pre_ms,load_ms,acc_ms,post_ms\n");
    }
    if (!dets_path.empty()) {
      dets = std::fopen(dets_path.c_str(), "w");
      if (!dets) throw std::runtime_error("impossible d'écrire " + dets_path);
    }

    std::printf("%s (%s) : images [%ld, %ld) de %s\n", m.network.c_str(), dev->name(), start,
                end, inputs.empty() ? images.c_str() : inputs.c_str());
    for (int w = 0; w < warmup && start < end; ++w) {  // caches, pages du u-dma-buf, IRQ
      fetch(start);
      forward();
    }

    PowerSampler power(power_path);
    double p_idle = 0.0;
    if (!power_path.empty()) {
      power.start();
      std::this_thread::sleep_for(std::chrono::duration<double>(idle_s));
      p_idle = power.stop();
      power.start();
    }

    std::vector<double> t_pre, t_load, t_acc, t_post, t_tot;
    const auto t_run = std::chrono::steady_clock::now();
    for (long k = start; k < end; ++k) {
      auto t0 = std::chrono::steady_clock::now();
      const std::string id = fetch(k);
      const double pre = sw::seconds_since(t0);
      t0 = std::chrono::steady_clock::now();
      acc.load_input(x.data());
      const double load = sw::seconds_since(t0);
      double a = 0.0;
      for (const driver::ConvCall& c : acc.program().calls) a += acc.run_layer(c);
      t0 = std::chrono::steady_clock::now();
      const auto d = hw_post ? sw::hw_detections(acc.run_post(conf, iou))
                             : sw::detect(m, acc.program(), acc.arena(), conf, iou);
      const double post = sw::seconds_since(t0);

      t_pre.push_back(1e3 * pre);
      t_load.push_back(1e3 * load);
      t_acc.push_back(1e3 * a);
      t_post.push_back(1e3 * post);
      t_tot.push_back(1e3 * (pre + load + a + post));
      if (times)
        std::fprintf(times, "%s,%.6f,%.6f,%.6f,%.6f\n", id.c_str(), 1e3 * pre, 1e3 * load,
                     1e3 * a, 1e3 * post);
      if (dets) {
        std::fprintf(dets, "%s\n",
                     golden::detections_json(d, "\"image\": \"" + id + "\", ").c_str());
        std::fflush(dets);
      }
      if ((k - start + 1) % 100 == 0) {
        std::printf("%ld/%ld images\n", k - start + 1, end - start);
        std::fflush(stdout);
      }
    }
    const double wall = sw::seconds_since(t_run);
    const double p_run = power_path.empty() ? 0.0 : power.stop();
    if (times) std::fclose(times);
    if (dets) std::fclose(dets);

    const size_t n = t_tot.size();
    std::printf("%zu images, %.1f s\n", n, wall);
    std::printf("%-6s %10s %10s\n", "étage", "moy. ms", "p99 ms");
    const char* names[] = {"pre", "load", "acc", "post", "total"};
    const std::vector<double>* cols[] = {&t_pre, &t_load, &t_acc, &t_post, &t_tot};
    for (int c = 0; c < 5; ++c)
      std::printf("%-6s %10.3f %10.3f\n", names[c], mean(*cols[c]), quantile(*cols[c], 0.99));
    if (n) std::printf("débit : %.2f images/s (passes enchaînées)\n", double(n) / wall);
    if (!power_path.empty())
      std::printf("puissance : repos %.3f W, en charge %.3f W (%ld échantillons)\n", p_idle,
                  p_run, power.samples());
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "erreur : %s\n", ex.what());
    return 1;
  }
}
