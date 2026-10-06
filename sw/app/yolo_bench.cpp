// Mesures sur l'ARM (T8.1, T8.2) : passe complète sur une suite d'images, temps par étage et
// par image, détections au seuil de la mAP, puissance de la carte.
//
//   yolo_bench --model DIR (--inputs inputs.bin --ids ids.txt | --images liste.txt)
//              [--start 0] [--count N] [--warmup 5] [--conf 0.005] [--iou 0.45]
//              [--times temps.csv] [--dets det.jsonl] [--hw-post] [--pipeline]
//              [--engine conv|stream]
//              [--power /sys/class/hwmon/hwmonN/power1_input] [--idle-s 5]
//              [--backend sim|uio] [--uio /dev/uioN] [--udmabuf udmabufN] [--poll] [--cached]
//
// --inputs : entrées int8 (C, H, W) concaténées, produites par tools/make_inputs.py avec le
// prétraitement du modèle entier Python : sortie comparable bit à bit (mAP au stade FPGA).
// Un `--inputs` en .npy (une ou quelques entrées, ex. dumps/<id>/input.npy) est accepté.
// --images : un chemin JPEG/PNG par ligne, prétraitement `stretch` sur l'ARM (preprocess.hpp)
// chronométré ; c'est le mode des mesures de latence (results/protocole.md).
//
// Étages (ms, par image) : pre = lecture de l'entrée (et décodage + redimensionnement +
// quantification en --images) ; load = mise à zéro de l'arène et copie de l'entrée en DDR ;
// acc = toutes les convs, enchaînées par le séquenceur du noyau (une seule fin à attendre,
// T10.7) ; post = décodage + NMS (sur l'ARM, ou par le noyau `yolo_post` avec --hw-post,
// T9.1) ; arm = temps CPU du thread pendant acc + post (attentes d'IRQ exclues).
// --pipeline (T10.5) : deux arènes, deux threads ; pre + load de l'image i + 1 recouvrent
// acc + post de l'image i. Le débit tend vers l'inverse du plus lent des deux étages au lieu
// de leur somme ; détections identiques au mode séquentiel.
// --engine stream (T10.9) : architecture streaming `yolo_stream` + AXI DMA (Tiny-YOLOv2) au
// lieu du moteur unique ; acc = transfert MM2S → noyau → S2MM de la tête ; post sur l'ARM.
// --hw-post : colonnes en plus dans --times, débordements de la sélection de `yolo_post` et,
// sur le backend sim, cycles estimés par la C-sim (total, NMS), candidates et survivantes
// (T11.6).
// --power : µW lus dans hwmon (INA260 du SOM KV260) toutes les 10 ms, au repos pendant
// --idle-s secondes puis pendant la boucle mesurée.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "accel_driver.hpp"
#include "stream_driver.hpp"
#include "golden/detections_io.hpp"
#include "golden/model.hpp"
#include "golden/npy.hpp"
#include "heads.hpp"
#include "options.hpp"
#include "preprocess.hpp"
#ifdef SW_HAVE_SIM
#include "postproc.hpp"
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace {

int usage() {
  std::fprintf(stderr,
               "usage : yolo_bench --model DIR (--inputs x.bin --ids ids.txt | --images l.txt)\n"
               "                   [--start 0] [--count N] [--warmup 5] [--conf 0.005]\n"
               "                   [--iou 0.45] [--times t.csv] [--dets d.jsonl] [--hw-post]\n"
               "                   [--pipeline] [--engine conv|stream]\n"
               "                   [--power .../power1_input] [--idle-s 5]\n"
               "                   [--backend sim|uio] [--uio /dev/uioN] [--udmabuf udmabufN] "
               "[--poll] [--cached]\n");
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
  std::string engine = "conv";
  double conf = 0.005, iou = 0.45, idle_s = 5.0;
  long start = 0, count = -1;
  int warmup = 5;
  bool hw_post = false, pipeline = false;
  for (int i = 1; i < argc; ++i) {
    if (sw::parse_device_option(argc, argv, i, dopt)) continue;
    const std::string k = argv[i];
    if (k == "--hw-post" || k == "--pipeline") {
      (k == "--pipeline" ? pipeline : hw_post) = true;
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
    else if (k == "--engine") engine = v;
    else return usage();
  }
  if (model_dir.empty() || inputs.empty() == images.empty()) return usage();
  if (!inputs.empty() && ids_path.empty()) return usage();
  if (engine != "conv" && engine != "stream") return usage();
  if (engine == "stream" && hw_post) {
    std::fprintf(stderr, "--hw-post : moteur conv seulement\n");
    return 2;
  }

  try {
    const golden::Model m = golden::Model::load(model_dir);
    const auto dev = driver::make_device(dopt);
    const int n_slots = pipeline ? 2 : 1;
    std::unique_ptr<driver::Accelerator> acc_p;
    std::unique_ptr<driver::StreamAccelerator> sacc;
    if (engine == "stream") sacc = std::make_unique<driver::StreamAccelerator>(*dev, m, n_slots);
    else acc_p = std::make_unique<driver::Accelerator>(*dev, m, n_slots);
    const size_t in_bytes = size_t(m.in_c) * m.in_h * m.in_w;
    if (!images.empty() && m.in_c != 1 && m.in_c != 3)
      throw std::runtime_error("entrée à 1 ou 3 canaux attendue");

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

    // Lecture (et prétraitement) de l'image k → entrée int8 `x` ; rend l'identifiant. Appelée
    // par un seul thread à la fois (le producteur en --pipeline).
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
      uint8_t* rgb = stbi_load(list[size_t(k)].c_str(), &W, &H, &n, m.in_c);
      if (!rgb) throw std::runtime_error("lecture impossible : " + list[size_t(k)]);
      x = sw::preprocess_stretch(rgb, W, H, m.in_c, m.in_h, m.in_w, m.input_scale);
      stbi_image_free(rgb);
      return stem(list[size_t(k)]);
    };
    // Moteur : unique (séquenceur, T10.7) ou streaming (T10.9).
    auto load = [&](int slot) {
      if (sacc) sacc->load_input(x.data(), slot);
      else acc_p->load_input(x.data(), slot);
    };
    auto run_acc = [&](int slot) { return sacc ? sacc->run(slot) : acc_p->run_all(slot); };
    auto forward = [&]() {
      load(0);
      return run_acc(0);
    };
    // Dernier appel de `yolo_post` (--hw-post) : débordements, puis compteurs de la C-sim.
#ifdef SW_HAVE_SIM
    const bool sim_cycles = std::string(dev->name()) == "sim";
#endif
    std::string post_stats;
    auto detect = [&](int slot) {
      if (sacc) return sw::detect(m, {sacc->head().data()}, conf, iou);
      if (!hw_post) return sw::detect(m, acc_p->program(), acc_p->arena(slot), conf, iou);
      int overflow = 0;
      const auto boxes = acc_p->run_post(conf, iou, &overflow, nullptr, slot);
      post_stats = "," + std::to_string(overflow);
#ifdef SW_HAVE_SIM
      if (sim_cycles) {
        const accel::PostCycles& c = accel::post_cycles;
        post_stats += "," + std::to_string(c.total) + "," + std::to_string(c.nms) + "," +
                      std::to_string(c.candidates) + "," + std::to_string(c.survivors);
      } else
#endif
        post_stats += ",,,,";
      return sw::hw_detections(boxes, hwpp::anchor_ref_w(m.in_h, m.in_w),
                               hwpp::anchor_ref_h(m.in_h, m.in_w));
    };

    std::FILE* times = nullptr;
    std::FILE* dets = nullptr;
    if (!times_path.empty()) {
      times = std::fopen(times_path.c_str(), "w");
      if (!times) throw std::runtime_error("impossible d'écrire " + times_path);
      std::fprintf(times, "id,pre_ms,load_ms,acc_ms,post_ms,arm_ms%s\n",
                   hw_post ? ",overflow,post_cycles,nms_cycles,candidates,survivors" : "");
    }
    if (!dets_path.empty()) {
      dets = std::fopen(dets_path.c_str(), "w");
      if (!dets) throw std::runtime_error("impossible d'écrire " + dets_path);
    }

    std::printf("%s (%s, %s%s) : images [%ld, %ld) de %s\n", m.network.c_str(), dev->name(),
                engine.c_str(), pipeline ? ", pipeline" : "", start, end,
                inputs.empty() ? images.c_str() : inputs.c_str());
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

    std::vector<double> t_pre, t_load, t_acc, t_post, t_arm, t_tot;
    // Étage 2 (thread principal) : accélérateur + post-traitement sur l'arène `slot`.
    auto finish = [&](long k, const std::string& id, int slot, double pre, double load) {
      const double cpu0 = sw::thread_cpu_seconds();
      const double a = run_acc(slot);
      auto t0 = std::chrono::steady_clock::now();
      const auto d = detect(slot);
      const double post = sw::seconds_since(t0);
      const double arm = sw::thread_cpu_seconds() - cpu0;

      t_pre.push_back(1e3 * pre);
      t_load.push_back(1e3 * load);
      t_acc.push_back(1e3 * a);
      t_post.push_back(1e3 * post);
      t_arm.push_back(1e3 * arm);
      t_tot.push_back(1e3 * (pre + load + a + post));
      if (times)
        std::fprintf(times, "%s,%.6f,%.6f,%.6f,%.6f,%.6f%s\n", id.c_str(), 1e3 * pre,
                     1e3 * load, 1e3 * a, 1e3 * post, 1e3 * arm, post_stats.c_str());
      if (dets) {
        std::fprintf(dets, "%s\n",
                     golden::detections_json(d, "\"image\": \"" + id + "\", ").c_str());
        std::fflush(dets);
      }
      if ((k - start + 1) % 100 == 0) {
        std::printf("%ld/%ld images\n", k - start + 1, end - start);
        std::fflush(stdout);
      }
    };
    // Étage 1 : lecture, prétraitement, copie dans l'arène `slot`.
    struct Ready {
      long k;
      std::string id;
      int slot;
      double pre, load;
    };
    auto prepare = [&](long k, int slot) {
      auto t0 = std::chrono::steady_clock::now();
      Ready r{k, fetch(k), slot, sw::seconds_since(t0), 0.0};
      t0 = std::chrono::steady_clock::now();
      load(slot);
      r.load = sw::seconds_since(t0);
      return r;
    };

    const auto t_run = std::chrono::steady_clock::now();
    if (!pipeline) {
      for (long k = start; k < end; ++k) {
        const Ready r = prepare(k, 0);
        finish(k, r.id, 0, r.pre, r.load);
      }
    } else {
      // Deux arènes : le producteur remplit une arène libre pendant que le thread principal
      // fait tourner le noyau sur l'autre ; images rendues dans l'ordre.
      std::mutex mu;
      std::condition_variable cv;
      std::deque<int> free_slots{0, 1};
      std::deque<Ready> ready;
      std::exception_ptr error;
      std::thread producer([&] {
        try {
          for (long k = start; k < end; ++k) {
            int slot;
            {
              std::unique_lock<std::mutex> lk(mu);
              cv.wait(lk, [&] { return !free_slots.empty() || error; });
              if (error) return;
              slot = free_slots.front();
              free_slots.pop_front();
            }
            Ready r = prepare(k, slot);
            {
              const std::lock_guard<std::mutex> lk(mu);
              ready.push_back(std::move(r));
            }
            cv.notify_all();
          }
        } catch (...) {
          const std::lock_guard<std::mutex> lk(mu);
          error = std::current_exception();
          cv.notify_all();
        }
      });
      try {
        for (long k = start; k < end; ++k) {
          Ready r;
          {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return !ready.empty() || error; });
            if (error) break;
            r = std::move(ready.front());
            ready.pop_front();
          }
          finish(r.k, r.id, r.slot, r.pre, r.load);
          {
            const std::lock_guard<std::mutex> lk(mu);
            free_slots.push_back(r.slot);
          }
          cv.notify_all();
        }
      } catch (...) {
        const std::lock_guard<std::mutex> lk(mu);
        error = std::current_exception();
        cv.notify_all();
      }
      producer.join();
      if (error) std::rethrow_exception(error);
    }
    const double wall = sw::seconds_since(t_run);
    const double p_run = power_path.empty() ? 0.0 : power.stop();
    if (times) std::fclose(times);
    if (dets) std::fclose(dets);

    const size_t n = t_tot.size();
    std::printf("%zu images, %.1f s\n", n, wall);
    std::printf("%-6s %10s %10s\n", "étage", "moy. ms", "p99 ms");
    const char* names[] = {"pre", "load", "acc", "post", "arm", "total"};
    const std::vector<double>* cols[] = {&t_pre, &t_load, &t_acc, &t_post, &t_arm, &t_tot};
    for (int c = 0; c < 6; ++c)
      std::printf("%-6s %10.3f %10.3f\n", names[c], mean(*cols[c]), quantile(*cols[c], 0.99));
    if (n) {
      // Étages du pipeline : (pre + load) et (acc + post).
      const double s1 = mean(t_pre) + mean(t_load), s2 = mean(t_acc) + mean(t_post);
      std::printf("débit : %.2f images/s mesuré (%s) ; attendu %.2f en séquentiel (somme des "
                  "étages), %.2f en pipeline (étage le plus lent)\n",
                  double(n) / wall, pipeline ? "pipeline" : "séquentiel", 1e3 / (s1 + s2),
                  1e3 / std::max(s1, s2));
    }
    if (!power_path.empty())
      std::printf("puissance : repos %.3f W, en charge %.3f W (%ld échantillons)\n", p_idle,
                  p_run, power.samples());
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "erreur : %s\n", ex.what());
    return 1;
  }
}
