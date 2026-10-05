// Options communes à yolo_app et run_compare : choix du backend matériel (sim | uio).
#pragma once

#include <chrono>
#include <string>

#include "device.hpp"

namespace sw {

// Consomme argv[i] (et sa valeur) si c'est une option de backend ; rend false sinon.
//   --backend sim|uio  --uio /dev/uioN  --udmabuf udmabufN  --poll (sans interruption)
//   --uio-post /dev/uioN (registres de yolo_post, T9.1)
inline bool parse_device_option(int argc, char** argv, int& i, driver::DeviceOptions& o) {
  const std::string k = argv[i];
  if (k == "--poll") {
    o.irq = false;
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

inline double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace sw
