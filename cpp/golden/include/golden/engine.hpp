// Réseau complet sur le moteur unique couche par couche (§10.1, §10.5 étape 3).
//
// La « DDR » est une arène par tampon nommé du manifest (`buffers`). Chaque couche a une
// **vue** de sortie (InView) :
// - conv : sa sortie dans la DDR (la carte poolée si le maxpool est fusionné, la vue du
//   maxpool ; la carte avant pooling est `prepool_out` ou, en débogage, une sonde) ;
// - upsample : la vue de sa source avec une division d'adresse par 2 — aucune copie (§10.3) ;
// - route : concaténation des segments de ses sources, déjà bout à bout en mémoire si le
//   placement le permet — aucune copie (§10.3) ;
// - yolo / region : la vue de la tête (le décodage est fait par l'hôte, postproc.hpp).
// Écart au manifest : la zone `out` de l'upsample (R:0, réservée pour une copie ×2) n'est
// jamais écrite ; la route 20 est lue en deux segments (L18 vue ×2, puis prépool de L08).
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "golden/conv.hpp"
#include "golden/model.hpp"
#include "golden/postproc.hpp"

namespace golden {

struct Tensor {
  int c = 0, h = 0, w = 0;
  std::vector<int8_t> data;
};

// Paramètres d'une conv du manifest ; `in_h`, `in_w` : forme de son entrée.
ConvParams conv_params(const Layer& l, int in_h, int in_w);
int prepool_h(const Layer& l);
int prepool_w(const Layer& l);

using DefaultTiles = Tiles<16, 16, 13, 13>;

class Engine {
 public:
  explicit Engine(const Model& m);

  // Passe avant complète. `input` : (C, H, W) int8. Si `record`, garde la sortie de chaque
  // couche (numérotation Lxx des dumps : carte avant pooling pour une conv fusionnée).
  template <class T = DefaultTiles>
  void run(const int8_t* input, bool record = false);

  const Model& model() const { return m_; }
  const InView& view(int id) const { return views_.at(size_t(id)); }
  Tensor materialize(const InView& v) const;
  const std::vector<Tensor>& recorded() const { return rec_; }
  const AccessStats& stats(int id) const { return stats_.at(size_t(id)); }
  std::vector<int8_t>& buffer(const std::string& name) { return arena_.at(name); }
  const std::vector<int8_t>& buffer(const std::string& name) const { return arena_.at(name); }

 private:
  int8_t* addr(const BufRef& r) { return arena_.at(r.buf).data() + r.offset; }
  InView in_view(const Layer& l) const;  // entrée d'une conv : vue de la couche précédente
  void place_views();                      // vues des couches sans calcul
  void record(int id);

  const Model& m_;
  std::map<std::string, std::vector<int8_t>> arena_;
  std::vector<InView> views_;
  std::vector<AccessStats> stats_;
  std::vector<std::vector<int8_t>> taps_;  // carte avant pooling (débogage) par conv
  std::vector<Tensor> rec_;
  bool recording_ = false;
};

// Têtes de sortie après `run`, prêtes pour postproc::postprocess (§9.4).
std::vector<postproc::Head> make_heads(const Engine& e);

template <class T>
void Engine::run(const int8_t* input, bool record_all) {
  recording_ = record_all;
  rec_.assign(m_.layers.size(), Tensor{});
  stats_.assign(m_.layers.size(), AccessStats{});
  const size_t n_in = size_t(m_.in_c) * m_.in_h * m_.in_w;
  std::copy(input, input + n_in, addr(m_.input));

  for (const Layer& l : m_.layers) {
    if (l.type == LayerType::Conv) {
      const InView in = in_view(l);
      const ConvParams p = conv_params(l, in.h, in.w);
      int8_t* pre = l.prepool_out.valid() ? addr(l.prepool_out) : nullptr;
      int8_t* tap = nullptr;
      if (recording_ && p.pooled()) {
        taps_[size_t(l.id)].resize(size_t(p.cout) * p.out_h() * p.out_w());
        tap = taps_[size_t(l.id)].data();
      }
      conv_layer<T>(p, in, m_.conv_weights(l), m_.conv_bias(l), m_.conv_m0(l), addr(l.out), pre,
                    tap, &stats_[size_t(l.id)]);
      if (recording_) {
        record(l.id);
        if (p.pooled()) record(l.pool_layer);
      }
    } else if (l.type == LayerType::Maxpool) {
      // Fusionné dans l'étage de sortie de la conv (enregistré avec elle).
    } else if (recording_) {
      record(l.id);  // vue sans calcul : la matérialiser tant que ses sources sont en place
    }
  }
}

}  // namespace golden
