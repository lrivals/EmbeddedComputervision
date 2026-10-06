// Post-traitement sur l'ARM (T7.3, §10.3 : décodage + NMS sur l'hôte comme 2025-kim et
// 2024-yan) : les têtes int8 sont lues en place dans l'arène DDR remplie par l'accélérateur,
// puis passées à `postproc::postprocess` (cpp/golden/include/golden/postproc.hpp, tel quel).
#pragma once

#include <vector>

#include "golden/hw_postproc.hpp"
#include "golden/model.hpp"
#include "golden/postproc.hpp"
#include "program.hpp"

namespace sw {

// Même construction que `golden::make_heads` (cpp/golden/src/engine.cpp), à partir des vues
// du driver. Les pointeurs renvoyés désignent `arena` et les tables de `m`.
std::vector<postproc::Head> make_heads(const golden::Model& m, const driver::Program& p,
                                       const int8_t* arena);

// Têtes lues à des adresses données (une par tête de `m.heads()`, (C, H, W) int8), par exemple
// la tête rendue par le streaming (T10.9).
std::vector<postproc::Head> make_heads(const golden::Model& m,
                                       const std::vector<const int8_t*>& data);

inline std::vector<postproc::Detection> detect(const golden::Model& m,
                                               const std::vector<const int8_t*>& data,
                                               double conf = 0.25, double iou = 0.45) {
  return postproc::postprocess(make_heads(m, data), conf, iou);
}

// Décodage + NMS aux seuils de la démo et des dumps (conf 0,25, IoU 0,45).
inline std::vector<postproc::Detection> detect(const golden::Model& m, const driver::Program& p,
                                               const int8_t* arena, double conf = 0.25,
                                               double iou = 0.45) {
  return postproc::postprocess(make_heads(m, p, arena), conf, iou);
}

// Boîtes entières de `yolo_post` (T9.1) → détections (cx, cy, w, h) normalisées, triées par
// score décroissant (stable) ; mêmes doubles que `to_detections` (hw_postproc.py).
// `ref_w` × `ref_h` : repère des coins (hwpp::anchor_ref_w / _h du modèle).
std::vector<postproc::Detection> hw_detections(const std::vector<hwpp::Box>& boxes,
                                               int ref_w = hwpp::ANCHOR_REF,
                                               int ref_h = hwpp::ANCHOR_REF);

}  // namespace sw
