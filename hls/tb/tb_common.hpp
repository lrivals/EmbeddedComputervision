// Outils communs aux testbenches HLS (C-sim g++ ou Vitis, co-sim).
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "golden/npy.hpp"

#ifndef ACC_MODEL_DIR
#define ACC_MODEL_DIR "../model"
#endif

namespace tb {

struct Args {
  std::string model_dir = ACC_MODEL_DIR;
  std::vector<std::string> nets;
  std::vector<std::string> images;
  int layer = -1;  // tb_conv : une seule conv
  std::string csv;
  int random = -1;  // tb_post : têtes aléatoires par réseau (-1 : défaut du banc)
};

inline const char* const ALL_NETS[] = {"tiny-yolov2-voc", "tiny-yolov3-coco"};
inline const char* const ALL_IMAGES[] = {"000001", "000002", "000003"};

// --model DIR  --net NAME (répétable)  --image ID (répétable)  --layer N  --csv FICHIER
// --random N
inline Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    if (i + 1 >= argc) {
      std::fprintf(stderr, "argument sans valeur : %s\n", k.c_str());
      std::exit(2);
    }
    const std::string v = argv[++i];
    if (k == "--model") a.model_dir = v;
    else if (k == "--net") a.nets.push_back(v);
    else if (k == "--image") a.images.push_back(v);
    else if (k == "--layer") a.layer = std::atoi(v.c_str());
    else if (k == "--csv") a.csv = v;
    else if (k == "--random") a.random = std::atoi(v.c_str());
    else {
      std::fprintf(stderr, "argument inconnu : %s\n", k.c_str());
      std::exit(2);
    }
  }
  if (a.nets.empty()) a.nets.assign(std::begin(ALL_NETS), std::end(ALL_NETS));
  if (a.images.empty()) a.images.assign(std::begin(ALL_IMAGES), std::end(ALL_IMAGES));
  return a;
}

inline std::string net_dir(const Args& a, const std::string& net) {
  return a.model_dir + "/" + net;
}

// Export absent : le testbench saute le réseau, sauf avec YOLO_REQUIRE_MODEL=1 (CI, T10.13)
// où il échoue, pour qu'une CI sans export ne soit pas verte à vide.
inline bool have_model(const Args& a, const std::string& net) {
  if (std::filesystem::exists(net_dir(a, net) + "/manifest.json")) return true;
  if (std::getenv("YOLO_REQUIRE_MODEL")) {
    std::fprintf(stderr, "export absent : %s (YOLO_REQUIRE_MODEL)\n", net_dir(a, net).c_str());
    std::exit(1);
  }
  return false;
}

// Dump Python : layer < 0 → input.npy, sinon Lxx.npy.
inline golden::NpyInt8 dump(const Args& a, const std::string& net, const std::string& image,
                            int layer) {
  char name[32];
  if (layer < 0) std::snprintf(name, sizeof name, "input.npy");
  else std::snprintf(name, sizeof name, "L%02d.npy", layer);
  return golden::npy_load_int8(net_dir(a, net) + "/dumps/" + image + "/" + name);
}

// Nombre de valeurs différentes ; affiche la première.
inline size_t compare(const char* what, const int8_t* got, const int8_t* want, size_t n) {
  size_t ndiff = 0;
  for (size_t k = 0; k < n; ++k)
    if (got[k] != want[k]) {
      if (ndiff == 0)
        std::printf("  ÉCART %s : indice %zu, noyau %d, attendu %d\n", what, k, int(got[k]),
                    int(want[k]));
      ++ndiff;
    }
  if (ndiff) std::printf("  %s : %zu / %zu valeurs différentes\n", what, ndiff, n);
  return ndiff;
}

}  // namespace tb
