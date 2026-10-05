// Modèle entier exporté (T4.7) : manifest.json + blobs — format de docs/conventions.md.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace golden {

enum class LayerType { Conv, Maxpool, Upsample, Route, Yolo, Region };

const char* layer_type_name(LayerType t);

// Tenseur (C, H, W) int8 contigu dans un tampon nommé, `offset` en octets.
struct BufRef {
  std::string buf;
  int64_t offset = -1;
  bool valid() const { return offset >= 0; }
};

struct Layer {
  int id = 0;
  LayerType type = LayerType::Conv;
  int out_c = 0, out_h = 0, out_w = 0;  // out_shape
  BufRef in, out;

  // conv
  int k = 0, s = 1, pad = 0, cin = 0, cout = 0;
  bool leaky = false;
  int pool_layer = -1, pool_k = 0, pool_s = 0;  // fused_pool (pool_layer < 0 : aucun)
  BufRef prepool_out;
  int64_t w_offset = 0, b_offset = 0, m0_offset = 0;
  int shift = 0;
  int qmax = 127;  // saturation de la sortie (activations à b bits : 2^{b−1} − 1, T9.3)
  double in_scale = 0.0, out_scale = 0.0;

  // maxpool
  int fused_into = -1;

  // route, upsample, têtes
  std::vector<int> from;
  std::string layout;
  double scale = 0.0;
  std::vector<int> mask;  // yolo
  int num = 0;            // region
  int64_t lut_offset = -1;
  int exp_frac = 0;

  bool is_head() const { return type == LayerType::Yolo || type == LayerType::Region; }
};

struct Model {
  int format_version = 0;
  std::string network;
  int classes = 0;
  int in_c = 0, in_h = 0, in_w = 0;
  double input_scale = 0.0;
  BufRef input;
  std::vector<std::pair<double, double>> anchors;  // pixels pour une entrée de 416
  std::map<std::string, int64_t> buffers;          // nom → taille en octets
  std::vector<Layer> layers;                       // indexées par id

  std::vector<int8_t> weights;    // weights.bin
  std::vector<int32_t> bias;      // bias.bin
  std::vector<int32_t> m0;        // requant.bin
  std::vector<uint32_t> luts;     // luts.bin

  // Lecture complète et contrôles (version, tailles de blobs, alignement 64, formes).
  static Model load(const std::string& dir);

  const int8_t* conv_weights(const Layer& l) const { return weights.data() + l.w_offset; }
  const int32_t* conv_bias(const Layer& l) const { return bias.data() + l.b_offset / 4; }
  const int32_t* conv_m0(const Layer& l) const { return m0.data() + l.m0_offset / 4; }
  // Tables d'une tête : σ, e^t, e^{d·s} (softmax v2), 256 uint32 chacune.
  const uint32_t* head_lut(const Layer& l, int which) const {
    return luts.data() + l.lut_offset / 4 + 256 * which;
  }
  std::vector<int> heads() const;
};

constexpr int64_t BLOB_ALIGN = 64;

}  // namespace golden
