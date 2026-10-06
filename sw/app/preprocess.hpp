// Prétraitement de la démo sur l'ARM (T7.3) : image 8 bits → entrée int8 (C, H, W).
//
// Redimensionnement `stretch` avec l'interpolation de Darknet (`resize_darknet`,
// python/yolo/infer/pipeline.py), même ordre d'opérations et mêmes types (pixels float32,
// coefficients float64, résultat float32), puis q = clip(⌊x / s + ½⌋, −127, 127)
// (`quantize_input`, python/yolo/quant/quantize.py). Les dumps de référence utilisent le
// letterbox PIL : cette entrée sert à la démo, pas à la comparaison bit-exacte.
// C = 3 (RGB) ou 1 (niveaux de gris, entrée thermique, T11.7) ; H ≠ W possible (T11.4).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace sw {

// `px` : (H, W, C) entrelacé ; sortie (C, out_h, out_w) int8.
inline std::vector<int8_t> preprocess_stretch(const uint8_t* px, int W, int H, int C,
                                              int out_h, int out_w, double input_scale) {
  struct Coord {
    int i, i1;
    double d;
  };
  auto coords = [](int n_out, int n_in) {
    std::vector<Coord> c(static_cast<size_t>(n_out));
    const double step = n_out > 1 ? double(n_in - 1) / (n_out - 1) : 0.0;
    for (int k = 0; k < n_out; ++k) {
      const double s = k * step;
      const int i = std::min(int(s), n_in - 1);
      c[size_t(k)] = {i, std::min(i + 1, n_in - 1), s - i};
    }
    return c;
  };
  const std::vector<Coord> cx = coords(out_w, W), cy = coords(out_h, H);
  auto pix = [&](int r, int q, int ch) {
    return float(px[(size_t(r) * W + q) * C + ch]) / 255.0f;
  };

  // Passe horizontale (H, out_w, C) en double, puis verticale.
  std::vector<double> part(size_t(H) * out_w * C);
  for (int r = 0; r < H; ++r)
    for (int k = 0; k < out_w; ++k)
      for (int ch = 0; ch < C; ++ch) {
        const Coord& c = cx[size_t(k)];
        part[(size_t(r) * out_w + k) * C + ch] =
            pix(r, c.i, ch) * (1 - c.d) + pix(r, c.i1, ch) * c.d;
      }
  std::vector<int8_t> out(size_t(C) * out_h * out_w);
  for (int k = 0; k < out_h; ++k)
    for (int q = 0; q < out_w; ++q)
      for (int ch = 0; ch < C; ++ch) {
        const Coord& c = cy[size_t(k)];
        const float x = float(part[(size_t(c.i) * out_w + q) * C + ch] * (1 - c.d) +
                              part[(size_t(c.i1) * out_w + q) * C + ch] * c.d);
        const double v = std::floor(double(x) / input_scale + 0.5);
        out[(size_t(ch) * out_h + k) * out_w + q] = int8_t(std::clamp(v, -127.0, 127.0));
      }
  return out;
}

}  // namespace sw
