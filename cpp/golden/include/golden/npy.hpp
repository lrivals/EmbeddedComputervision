// Lecture et écriture de tableaux .npy int8 (dumps du modèle entier, T4.7).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace golden {

struct NpyInt8 {
  std::vector<int64_t> shape;
  std::vector<int8_t> data;  // ordre C

  size_t numel() const;
};

NpyInt8 npy_load_int8(const std::string& path);
void npy_save_int8(const std::string& path, const std::vector<int64_t>& shape,
                   const int8_t* data);

}  // namespace golden
