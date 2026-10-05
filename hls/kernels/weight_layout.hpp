// Poids dans l'ordre des tuiles (T10.1) : le driver réordonne weights.bin une fois pour que la
// tuile de poids de chaque (to, ti) soit un bloc contigu, lu en mots m_axi pleins.
//
// Couche : blocs to → ti (ordre de parcours du noyau). Bloc (to, ti) de n voies (n = Tn, ou
// les voies valides restantes avec TRIM) : octets (i, j, tii, too), too le plus rapide, pour
// que WORD octets consécutifs tombent dans WORD bancs `too` différents de w_buf. Canaux de
// sortie complétés par des zéros jusqu'à Tm ; sans TRIM, voies complétées jusqu'à Tn.
// Voie = canal d'entrée, ou (canal, ligne du noyau) pour une conv pliée (fold).
// Poids 4 bits (d.wbits = 4, T10.10) : même ordre, deux poids par octet (quartet bas = too
// pair) ; Tm pair, donc un bloc fait exactement la moitié des octets.
//
// C++ simple : partagé par le noyau, le driver ARM et les testbenchs.
#pragma once

#include <cstdint>
#include <vector>

#include "accel_config.hpp"
#include "layer_desc.hpp"

namespace accel {

inline int lanes(const LayerDesc& d) { return d.fold ? d.cin * d.k : d.cin; }
inline int kern_h(const LayerDesc& d) { return d.fold ? 1 : d.k; }  // lignes du noyau calculées
inline int n_ti(const LayerDesc& d) { return cdiv(lanes(d), TN); }
// Voies chargées dans la tuile ti numéro `k`.
inline int ti_lanes(const LayerDesc& d, int k) {
  const int rest = lanes(d) - k * TN;
  return TRIM ? (rest < TN ? rest : TN) : TN;
}
// Poids d'un bloc de n voies, et ses octets en DDR.
inline int64_t w_block_elems(const LayerDesc& d, int n) {
  return int64_t(TM) * n * kern_h(d) * d.k;
}
inline int64_t w_block_bytes(const LayerDesc& d, int n) {
  return d.wbits == 4 ? w_block_elems(d, n) / 2 : w_block_elems(d, n);
}
// Octets d'un groupe de Tm canaux de sortie (tous ses blocs ti).
inline int64_t w_per_to(const LayerDesc& d) {
  return TRIM ? w_block_bytes(d, lanes(d)) : w_block_bytes(d, TN) * n_ti(d);
}
// Début du bloc (groupe de sortie im, tuile ti k), relatif à d.w_off.
inline int64_t w_block_off(const LayerDesc& d, int im, int k) {
  return int64_t(im) * w_per_to(d) + int64_t(k) * w_block_bytes(d, TN);
}
inline int64_t w_layer_bytes(const LayerDesc& d) { return cdiv(d.cout, TM) * w_per_to(d); }

// Réordonne les poids (cout, cin, k, k) d'une conv et les ajoute à `out` (d.w_off ignoré).
inline void reorder_weights(const LayerDesc& d, const int8_t* w, std::vector<int8_t>& out) {
  const int K = d.k, kh = kern_h(d), L = lanes(d);
  const size_t base = out.size();
  out.resize(base + size_t(w_layer_bytes(d)), 0);
  for (int im = 0; im < cdiv(d.cout, TM); ++im)
    for (int k = 0; k < n_ti(d); ++k) {
      const int n = ti_lanes(d, k);
      int8_t* blk = out.data() + base + w_block_off(d, im, k);
      for (int i = 0; i < kh; ++i)
        for (int j = 0; j < K; ++j)
          for (int tii = 0; tii < n; ++tii)
            for (int too = 0; too < TM; ++too) {
              const int o = im * TM + too, lane = k * TN + tii;
              if (o >= d.cout || lane >= L) continue;
              const int c = d.fold ? lane / K : lane, r = d.fold ? lane % K : i;
              const int8_t v = w[((o * d.cin + c) * K + r) * K + j];
              const int64_t e = ((i * K + j) * n + tii) * TM + too;
              if (d.wbits == 4)  // too pair : quartet bas ; impair : quartet haut
                blk[e / 2] = int8_t((uint8_t(blk[e / 2]) & (e % 2 ? 0x0F : 0xF0)) |
                                    ((uint8_t(v) & 0x0F) << (e % 2 ? 4 : 0)));
              else
                blk[e] = v;
            }
    }
}

}  // namespace accel
