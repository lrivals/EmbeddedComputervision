// detections.json : écriture (golden_run, sw/app) et lecture pour comparaison exacte
// (T5.4, T7.3). Doubles en %.17g : aller-retour exact avec le `json.dumps` Python.
#pragma once

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "golden/json.hpp"
#include "golden/postproc.hpp"

namespace golden {

// Objet JSON {"boxes", "scores", "labels"} sur une ligne ; `extra` (ex. `"image": "000001", `)
// est inséré en tête tel quel.
inline std::string detections_json(const std::vector<postproc::Detection>& d,
                                   const std::string& extra = "") {
  std::string s = "{" + extra + "\"boxes\": [";
  char buf[128];
  for (size_t k = 0; k < d.size(); ++k) {
    std::snprintf(buf, sizeof buf, "%s[%.17g, %.17g, %.17g, %.17g]", k ? ", " : "", d[k].box[0],
                  d[k].box[1], d[k].box[2], d[k].box[3]);
    s += buf;
  }
  s += "], \"scores\": [";
  for (size_t k = 0; k < d.size(); ++k) {
    std::snprintf(buf, sizeof buf, "%s%.17g", k ? ", " : "", d[k].score);
    s += buf;
  }
  s += "], \"labels\": [";
  for (size_t k = 0; k < d.size(); ++k) s += (k ? ", " : "") + std::to_string(d[k].label);
  return s + "]}";
}

inline void write_detections(const std::string& path,
                             const std::vector<postproc::Detection>& d) {
  FILE* f = std::fopen(path.c_str(), "w");
  if (!f) throw std::runtime_error("impossible d'écrire " + path);
  std::fprintf(f, "%s\n", detections_json(d).c_str());
  std::fclose(f);
}

// Champs `boxes`, `scores`, `labels` (les autres, comme `image` ou `conf`, sont ignorés).
inline std::vector<postproc::Detection> read_detections(const std::string& path) {
  const Json j = Json::parse_file(path);
  const Json &boxes = j["boxes"], &scores = j["scores"], &labels = j["labels"];
  if (boxes.size() != scores.size() || boxes.size() != labels.size())
    throw std::runtime_error(path + " : tailles incohérentes");
  std::vector<postproc::Detection> d(boxes.size());
  for (size_t k = 0; k < d.size(); ++k) {
    for (size_t q = 0; q < 4; ++q) d[k].box[q] = boxes[k][q].num();
    d[k].score = scores[k].num();
    d[k].label = int(labels[k].integer());
  }
  return d;
}

// Égalité exacte (contrat bit-exact, docs/conventions.md).
inline bool same_detections(const std::vector<postproc::Detection>& a,
                            const std::vector<postproc::Detection>& b) {
  if (a.size() != b.size()) return false;
  for (size_t k = 0; k < a.size(); ++k) {
    if (a[k].score != b[k].score || a[k].label != b[k].label) return false;
    for (int q = 0; q < 4; ++q)
      if (a[k].box[q] != b[k].box[q]) return false;
  }
  return true;
}

}  // namespace golden
