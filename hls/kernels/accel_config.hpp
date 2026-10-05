// Configuration de compilation du noyau `yolo_conv` (C++ simple, sans ap_int) : partagée par
// le noyau, le driver ARM (ordre des poids, choix des tuiles) et tools/perf_model.py, qui lit
// les valeurs par défaut de ce fichier.
//
// Tuiles : hls/configs/<carte>.tcl → -DACC_TM=…, défauts KV260 (ADR 0003). Les autres options
// sont les pistes de M10 ; `-DACC_WORD_BYTES=1 -DACC_TRIM=0 -DACC_RQ=1 -DACC_FOLD=0
// -DACC_TILE_POOL=0` redonne les cycles du noyau M6.
#pragma once

#ifndef ACC_TM
#define ACC_TM 32
#endif
#ifndef ACC_TN
#define ACC_TN 24
#endif
#ifndef ACC_TR
#define ACC_TR 13
#endif
#ifndef ACC_TC
#define ACC_TC 13
#endif
// Octets par mot des ports m_axi act_in, act_out, wts (T10.1) : 1, 2, 4 ou 8.
#ifndef ACC_WORD_BYTES
#define ACC_WORD_BYTES 8
#endif
// Ne charger que les canaux d'entrée valides de la dernière tuile ti (T10.3).
#ifndef ACC_TRIM
#define ACC_TRIM 1
#endif
// Canaux requantifiés par cycle dans l'étage de sortie (T10.2).
#ifndef ACC_RQ
#define ACC_RQ 8
#endif
// Pliage d'une conv à cin·k ≤ Tn (L00) : voies = (canal, ligne du noyau), conv 1 × k (T10.4).
#ifndef ACC_FOLD
#define ACC_FOLD 1
#endif
// Tr = Tc des convs suivies d'un maxpool 2×2 de stride 2 ; 0 : Tr, Tc (T10.4).
#ifndef ACC_TILE_POOL
#define ACC_TILE_POOL 14
#endif

namespace accel {

constexpr int TM = ACC_TM, TN = ACC_TN, TR = ACC_TR, TC = ACC_TC;
constexpr int WORD = ACC_WORD_BYTES;
constexpr bool TRIM = ACC_TRIM != 0;
constexpr int RQ = ACC_RQ;
constexpr bool FOLD = ACC_FOLD != 0;
constexpr int TILE_POOL = ACC_TILE_POOL;
constexpr int K_MAX = 3;
// Tampons sur puce : la plus grande des tuiles utilisées.
constexpr int TRB = TILE_POOL > TR ? TILE_POOL : TR;
constexpr int TCB = TILE_POOL > TC ? TILE_POOL : TC;
constexpr int IR = TRB + K_MAX - 1;  // S = 1 : S·Tr + K − S
constexpr int IC = TCB + K_MAX - 1;

static_assert(WORD == 1 || WORD == 2 || WORD == 4 || WORD == 8, "mot de 1, 2, 4 ou 8 octets");
static_assert(TM % (2 * WORD) == 0,
              "blocs de poids alignés sur un mot, y compris paquetés en 4 bits : Tm multiple de 2·WORD");
static_assert(RQ >= 1 && TM % RQ == 0, "Tm multiple de RQ");
static_assert(TILE_POOL == 0 || TILE_POOL % 2 == 0, "tuile poolée paire (maxpool stride 2)");

inline int cdiv(int a, int b) { return (a + b - 1) / b; }
// Mots couvrant n octets consécutifs, quel que soit leur alignement.
inline int row_words(int n) { return cdiv(n + WORD - 1, WORD); }

}  // namespace accel
