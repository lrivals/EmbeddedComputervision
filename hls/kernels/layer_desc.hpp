// Paramètres d'une couche conv du noyau `yolo_conv` (registres s_axilite, T6.2).
//
// C++ simple, sans ap_int : partagé par le noyau HLS, le testbench et le driver ARM
// (sw/driver, M7). Toutes les adresses sont des **indices** dans les tableaux passés en m_axi :
// - activations : octets de l'arène DDR contiguë (tampons nommés du manifest bout à bout) ;
// - poids : octets de weights.bin ;
// - paramètres : int32 de bias.bin suivi de requant.bin.
#pragma once

#include <cstdint>

namespace accel {

constexpr int MAX_SEGMENTS = 2;  // route 20 de Tiny-YOLOv3 : L19 (vue ×2) puis prépool de L08

// Entrée logique = concaténation sur les canaux de 1 ou 2 segments (route par adressage,
// §10.3) ; un segment est lu avec une division d'adresse par 2^up (upsample ×2 sans copie).
// Champs à plat (pas de tableau) : chaque champ devient un registre s_axilite.
struct LayerDesc {
  int32_t k, pad;          // noyau, complétion (stride 1)
  int32_t cin, cout;       // canaux
  int32_t h, w;            // forme logique de l'entrée
  int32_t shift;           // n de la requantification
  int32_t leaky;           // 0 / 1
  int32_t pool_k, pool_s;  // maxpool fusionné ; 0 : aucun
  int32_t nseg;            // 1 ou 2
  int32_t seg0_off, seg0_c, seg0_h, seg0_w, seg0_up;
  int32_t seg1_off, seg1_c, seg1_h, seg1_w, seg1_up;
  int32_t out_off;         // carte poolée (ou sortie conv sans pooling)
  int32_t prepool_off;     // carte avant pooling en DDR ; −1 : non écrite
  int32_t w_off;           // octets dans weights.bin
  int32_t b_off, m0_off;   // indices int32 dans le tableau de paramètres
  int32_t qmax;            // saturation de la sortie (127 ; 2^{b−1} − 1 à b bits, T9.3)

  bool pooled() const { return pool_k > 0; }
  int pk() const { return pooled() ? pool_k : 1; }
  int ps() const { return pooled() ? pool_s : 1; }
  int out_h() const { return h + 2 * pad - k + 1; }
  int out_w() const { return w + 2 * pad - k + 1; }
  // §4.4 : stride 1 → complétion par réplication, la taille est conservée.
  int pool_h() const { return ps() == 1 ? out_h() : (out_h() - pk()) / ps() + 1; }
  int pool_w() const { return ps() == 1 ? out_w() : (out_w() - pk()) / ps() + 1; }
};

}  // namespace accel
