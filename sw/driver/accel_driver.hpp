// Accélérateur piloté par l'ARM (T7.2) : programme du manifest (program.hpp) exécuté par le
// noyau `yolo_conv` à travers ses registres AXI-Lite.
//
// Mémoire : un seul tampon contigu, découpé en [arène | weights.bin | paramètres], chaque
// partie alignée à 64 octets. L'arène est celle de `driver::build` (tampons du manifest bout
// à bout), donc le placement contigu de la route est conservé ; les offsets des `LayerDesc`
// sont des indices relatifs aux trois pointeurs m_axi.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>

#include "device.hpp"
#include "golden/model.hpp"
#include "program.hpp"

namespace driver {

class Accelerator {
 public:
  Accelerator(Device& dev, const golden::Model& m);

  const Program& program() const { return prog_; }
  const golden::Model& model() const { return m_; }
  const int8_t* arena() const { return reinterpret_cast<const int8_t*>(mem_.virt); }
  int8_t* arena() { return reinterpret_cast<int8_t*>(mem_.virt); }

  // Remet l'arène à zéro et y copie l'entrée (C, H, W) int8.
  void load_input(const int8_t* input);
  // Une conv (+ maxpool fusionné) ; rend la durée en secondes (registres + calcul).
  double run_layer(const ConvCall& c);
  // Passe avant complète ; `after(call, secondes)` après chaque conv.
  void run(const int8_t* input,
           const std::function<void(const ConvCall&, double)>& after = nullptr);

 private:
  Device& dev_;
  const golden::Model& m_;
  Program prog_;
  Buffer mem_;
  uint64_t w_phys_ = 0, p_phys_ = 0;
};

}  // namespace driver
