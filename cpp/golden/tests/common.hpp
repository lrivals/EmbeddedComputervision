// Accès aux exports réels model/<net>/ pour les tests (sautés s'ils manquent).
#pragma once

#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <filesystem>
#include <map>
#include <memory>
#include <string>

#include "golden/model.hpp"
#include "golden/npy.hpp"

namespace testing {

inline const char* const NETS[] = {"tiny-yolov2-voc", "tiny-yolov3-coco"};
inline const char* const IMAGES[] = {"000001", "000002", "000003"};

inline std::string model_dir(const std::string& net) {
  return std::string(GOLDEN_MODEL_DIR) + "/" + net;
}

inline bool have_model(const std::string& net) {
  return std::filesystem::exists(model_dir(net) + "/manifest.json");
}

// Modèle chargé une fois par réseau.
inline const golden::Model& model(const char* net) {
  static std::map<std::string, std::unique_ptr<golden::Model>> cache;
  auto& m = cache[net];
  if (!m) m = std::make_unique<golden::Model>(golden::Model::load(model_dir(net)));
  return *m;
}

// Dump Python : "input" ou numéro de couche.
inline golden::NpyInt8 dump(const std::string& net, const std::string& image, int layer) {
  char name[32];
  if (layer < 0) std::snprintf(name, sizeof name, "input.npy");
  else std::snprintf(name, sizeof name, "L%02d.npy", layer);
  return golden::npy_load_int8(model_dir(net) + "/dumps/" + image + "/" + name);
}

}  // namespace testing

#define REQUIRE_MODEL(net) \
  if (!testing::have_model(net)) SKIP("export absent : make export (T4.7)")
