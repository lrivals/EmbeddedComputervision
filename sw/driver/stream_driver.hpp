// Architecture streaming pilotée par l'ARM (T10.9) : `yolo_stream` reçoit l'entrée HWC par un
// AXI DMA (MM2S) et rend la tête HWC par le même DMA (S2MM, jusqu'à TLAST).
//
// Mémoire : un tampon contigu [entrée 0 | … | entrée n−1 | tête | poids | biais | M0], chaque
// partie alignée à 64 octets. Les poids sont weights.bin dépaquetés (golden::Model) : en
// matériel seuls ceux des étages FRAME (L12, L13) sont lus, les autres sont en ROM ; la C-sim
// sans ROM lit tout. Plusieurs entrées (T10.5) : l'ARM prépare l'image suivante pendant que
// le DMA envoie la courante.
#pragma once

#include <cstdint>
#include <vector>

#include "device.hpp"
#include "golden/model.hpp"
#include "stream_desc.hpp"

namespace driver {

// Registres de `yolo_stream` pour Tiny-YOLOv2 : étages = convs de stream::STAGE_LAYER.
stream::StreamDesc stream_desc(const golden::Model& m);

class StreamAccelerator {
 public:
  StreamAccelerator(Device& dev, const golden::Model& m, int n_inputs = 1);

  // Entrée (C, H, W) int8 → HWC dans l'emplacement `slot`.
  void load_input(const int8_t* chw, int slot = 0);
  // Une image : ap_start, S2MM armé, MM2S lancé, attente de la fin du S2MM ; rend la durée (s).
  double run(int slot = 0);
  // Tête (C, H, W) int8 de la dernière image, convertie depuis le HWC reçu.
  const std::vector<int8_t>& head() const { return head_; }

 private:
  Device& dev_;
  const golden::Model& m_;
  int n_inputs_;
  int head_id_;
  Buffer mem_;
  int64_t in_stride_ = 0, out_at_ = 0, w_at_ = 0, b_at_ = 0, m0_at_ = 0;
  std::vector<int8_t> head_;
};

}  // namespace driver
