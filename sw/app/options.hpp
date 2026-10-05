// Options communes à yolo_app et run_compare : choix du backend matériel (sim | uio).
#pragma once

#include <chrono>
#include <ctime>
#include <string>

#include "device.hpp"

namespace sw {

// Consomme argv[i] (et sa valeur) si c'est une option de backend ; rend false sinon.
//   --backend sim|uio  --uio /dev/uioN  --udmabuf udmabufN  --poll (sans interruption)
//   --uio-post /dev/uioN (registres de yolo_post, T9.1)
//   --cached (u-dma-buf caché, synchronisations explicites, T10.6)
inline bool parse_device_option(int argc, char** argv, int& i, driver::DeviceOptions& o) {
  const std::string k = argv[i];
  if (k == "--poll") {
    o.irq = false;
    return true;
  }
  if (k == "--cached") {
    o.cached = true;
    return true;
  }
  std::string* dst = k == "--backend" ? &o.backend
                     : k == "--uio"   ? &o.uio
                     : k == "--udmabuf" ? &o.udmabuf
                     : k == "--uio-post" ? &o.uio_post
                                        : nullptr;
  if (!dst || i + 1 >= argc) return false;
  *dst = argv[++i];
  return true;
}

// Temps CPU du thread appelant (s) : temps ARM réellement occupé, attente d'IRQ exclue.
inline double thread_cpu_seconds() {
  timespec ts{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return double(ts.tv_sec) + 1e-9 * double(ts.tv_nsec);
}

inline double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace sw
