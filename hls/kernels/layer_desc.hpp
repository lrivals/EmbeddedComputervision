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
  int32_t tr, tc;          // tuile de sortie (avant pooling) de cette couche (T10.4)
  int32_t fold;            // 0 / 1 : voies = (canal, ligne du noyau), conv 1 × k (T10.4)

  bool pooled() const { return pool_k > 0; }
  int pk() const { return pooled() ? pool_k : 1; }
  int ps() const { return pooled() ? pool_s : 1; }
  int out_h() const { return h + 2 * pad - k + 1; }
  int out_w() const { return w + 2 * pad - k + 1; }
  // §4.4 : stride 1 → complétion par réplication, la taille est conservée.
  int pool_h() const { return ps() == 1 ? out_h() : (out_h() - pk()) / ps() + 1; }
  int pool_w() const { return ps() == 1 ? out_w() : (out_w() - pk()) / ps() + 1; }
};

constexpr int D_WORDS = 30;
static_assert(sizeof(LayerDesc) == 4 * D_WORDS, "LayerDesc : 30 × int32");

// Descripteur lu en DDR par le séquenceur (T10.7) : champ i au mot i, comme les registres.
inline LayerDesc desc_from_words(const int32_t w[D_WORDS]) {
  LayerDesc d;
  d.k = w[0];
  d.pad = w[1];
  d.cin = w[2];
  d.cout = w[3];
  d.h = w[4];
  d.w = w[5];
  d.shift = w[6];
  d.leaky = w[7];
  d.pool_k = w[8];
  d.pool_s = w[9];
  d.nseg = w[10];
  d.seg0_off = w[11];
  d.seg0_c = w[12];
  d.seg0_h = w[13];
  d.seg0_w = w[14];
  d.seg0_up = w[15];
  d.seg1_off = w[16];
  d.seg1_c = w[17];
  d.seg1_h = w[18];
  d.seg1_w = w[19];
  d.seg1_up = w[20];
  d.out_off = w[21];
  d.prepool_off = w[22];
  d.w_off = w[23];
  d.b_off = w[24];
  d.m0_off = w[25];
  d.qmax = w[26];
  d.tr = w[27];
  d.tc = w[28];
  d.fold = w[29];
  return d;
}

}  // namespace accel
