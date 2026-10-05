// Accélérateur piloté par l'ARM (T7.2) : programme du manifest (program.hpp) exécuté par le
// noyau `yolo_conv` à travers ses registres AXI-Lite.
//
// Mémoire : un seul tampon contigu, découpé en [arène 0 | … | arène n−1 | poids | paramètres |
// descripteurs], chaque partie alignée à 64 octets. Une arène est celle de `driver::build`
// (tampons du manifest bout à bout, marge comprise), donc le placement contigu de la route est
// conservé ; les offsets des `LayerDesc` sont des indices relatifs aux pointeurs m_axi, si bien
// que toutes les arènes partagent les mêmes descripteurs. Plusieurs arènes (T10.5) : l'ARM
// prépare l'entrée de l'image suivante pendant que le noyau calcule dans une autre.
// Poids dans l'ordre des tuiles (Program::weights) ; descripteurs : table du séquenceur
// (T10.7). Si le périphérique a le noyau `yolo_post` (T9.1), deux zones suivent : sa table
// (tables + descripteurs) et son résultat.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>

#include "device.hpp"
#include "golden/hw_postproc.hpp"
#include "golden/model.hpp"
#include "postproc.hpp"
#include "program.hpp"

namespace driver {

class Accelerator {
 public:
  Accelerator(Device& dev, const golden::Model& m, int n_arenas = 1);

  const Program& program() const { return prog_; }
  const golden::Model& model() const { return m_; }
  int n_arenas() const { return n_arenas_; }
  const int8_t* arena(int slot = 0) const {
    return reinterpret_cast<const int8_t*>(mem_.virt) + slot * arena_stride_;
  }
  int8_t* arena(int slot = 0) {
    return reinterpret_cast<int8_t*>(mem_.virt) + slot * arena_stride_;
  }

  // Remet l'arène `slot` à zéro et y copie l'entrée (C, H, W) int8. Peut tourner dans un autre
  // thread que les appels du noyau, sur une autre arène.
  void load_input(const int8_t* input, int slot = 0);
  // Une conv (+ maxpool fusionné) ; rend la durée en secondes (registres + calcul).
  double run_layer(const ConvCall& c, int slot = 0);
  // Toutes les convs par le séquenceur du noyau (T10.7) : une seule fin à attendre.
  double run_all(int slot = 0);
  // Passe avant complète, couche par couche ; `after(call, secondes)` après chaque conv.
  void run(const int8_t* input,
           const std::function<void(const ConvCall&, double)>& after = nullptr);
  // Post-traitement matériel sur les têtes de l'arène `slot` (T9.1) : boîtes de `yolo_post`
  // (coins Q4, scores Q16). `seconds` : registres + calcul + lecture du résultat.
  std::vector<hwpp::Box> run_post(double conf, double iou, int* overflow = nullptr,
                                  double* seconds = nullptr, int slot = 0);

 private:
  uint64_t arena_phys(int slot) const { return mem_.phys + uint64_t(slot * arena_stride_); }
  void set_ptrs(int slot);
  void start_and_wait(int slot);

  Device& dev_;
  const golden::Model& m_;
  Program prog_;
  Buffer mem_;
  int n_arenas_ = 1;
  int64_t arena_stride_ = 0;
  uint64_t w_phys_ = 0, p_phys_ = 0, desc_phys_ = 0;
  int64_t tab_at_ = 0, res_at_ = 0;  // octets dans mem_ (yolo_post)
  double post_conf_ = -1.0, post_iou_ = -1.0;
  std::vector<int64_t> head_off_;
  accel::PostDesc post_desc_{};
};

}  // namespace driver
