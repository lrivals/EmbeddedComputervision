// Prétraitement de la démo sur l'ARM (T7.3) : image RGB 8 bits → entrée int8 (3, S, S).
//
// Redimensionnement `stretch` avec l'interpolation de Darknet (`resize_darknet`,
// python/yolo/infer/pipeline.py), même ordre d'opérations et mêmes types (pixels float32,
// coefficients float64, résultat float32), puis q = clip(⌊x / s + ½⌋, −127, 127)
// (`quantize_input`, python/yolo/quant/quantize.py). Les dumps de référence utilisent le
// letterbox PIL : cette entrée sert à la démo, pas à la comparaison bit-exacte.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace sw {

// `rgb` : (H, W, 3) entrelacé ; sortie (3, size, size) int8.
inline std::vector<int8_t> preprocess_stretch(const uint8_t* rgb, int W, int H, int size,
                                              double input_scale) {
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
  const std::vector<Coord> cx = coords(size, W), cy = coords(size, H);
  auto px = [&](int r, int q, int ch) { return float(rgb[(size_t(r) * W + q) * 3 + ch]) / 255.0f; };

  // Passe horizontale (H, size, 3) en double, puis verticale.
  std::vector<double> part(size_t(H) * size * 3);
  for (int r = 0; r < H; ++r)
    for (int k = 0; k < size; ++k)
      for (int ch = 0; ch < 3; ++ch) {
        const Coord& c = cx[size_t(k)];
        part[(size_t(r) * size + k) * 3 + ch] =
            px(r, c.i, ch) * (1 - c.d) + px(r, c.i1, ch) * c.d;
      }
  std::vector<int8_t> out(size_t(3) * size * size);
  for (int k = 0; k < size; ++k)
    for (int q = 0; q < size; ++q)
      for (int ch = 0; ch < 3; ++ch) {
        const Coord& c = cy[size_t(k)];
        const float x = float(part[(size_t(c.i) * size + q) * 3 + ch] * (1 - c.d) +
                              part[(size_t(c.i1) * size + q) * 3 + ch] * c.d);
        const double v = std::floor(double(x) / input_scale + 0.5);
        out[(size_t(ch) * size + k) * size + q] = int8_t(std::clamp(v, -127.0, 127.0));
      }
  return out;
}

}  // namespace sw
