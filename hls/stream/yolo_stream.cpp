// Top streaming de Tiny-YOLOv2 (T9.4.2, T10.8). Voir yolo_stream.hpp.
#include "yolo_stream.hpp"

#ifdef STREAM_ROM
#include "stream_rom.hpp"  // généré : stream_rom::S0 … S5, S8 et STREAM_ROM_PRAGMAS
#endif

namespace stream {

#ifndef __SYNTHESIS__
StageCycles stream_cycles[N_STAGES];
bool stream_tap = false;
std::vector<int8_t> stream_taps[N_STAGES];

// Copie le contenu d'un flux (C-sim : l'étage précédent a produit toute sa sortie).
static void tap(int k, hls::stream<int8_t>& s) {
  if (!stream_tap) return;
  hls::stream<int8_t> keep;
  stream_taps[k].clear();
  while (!s.empty()) {
    const int8_t v = s.read();
    stream_taps[k].push_back(v);
    keep.write(v);
  }
  while (!keep.empty()) s.write(keep.read());
}
#define STAGE_ARGS(k) , stream_cycles[k]
#define TAP(k, s) tap(k, s)
#else
#define STAGE_ARGS(k)
#define TAP(k, s)
#endif

static StageParams params(const int8_t* wts, const int32_t* bias, const int32_t* m0,
                          const StreamDesc& d, int k, bool leaky) {
  StageParams p;
  p.w = wts + d.w_off[k];
  p.bias = bias + d.b_off[k];
  p.m0 = m0 + d.m0_off[k];
  p.shift = d.shift[k];
  p.qmax = d.qmax[k];
  p.leaky = leaky;
  return p;
}

static void from_axis(hls::stream<axis_byte>& in, hls::stream<int8_t>& s) {
  for (int i = 0; i < IN_VALUES; ++i) {
#pragma HLS PIPELINE II=1
    s.write(int8_t(in.read().data));
  }
}

static void to_axis(hls::stream<int8_t>& s, hls::stream<axis_byte>& out) {
  for (int i = 0; i < OUT_VALUES; ++i) {
#pragma HLS PIPELINE II=1
    axis_byte v;
    v.data = s.read();
    v.last = i == OUT_VALUES - 1;
#if defined(__SYNTHESIS__) || defined(__VITIS_HLS__)
    v.keep = -1;
    v.strb = -1;
#endif
    out.write(v);
  }
}

}  // namespace stream

using namespace stream;

// Paramètres de template d'un étage k : forme, puis repliement et modes des tables.
#define STAGE(k, K, CIN, COUT, H, W, PK, PS)                                               \
  conv_stage<K, CIN, COUT, H, W, STAGE_PE[k], STAGE_SIMD[k], PK, PS, STAGE_FRAME[k], ROM_##k, \
             (k > 0 && STAGE_OUT_CHW[k > 0 ? k - 1 : 0]), STAGE_OUT_CHW[k]>

// Poids des étages sur la puce : ROM générée, sinon weights.bin (C-sim de référence).
#if defined(STREAM_ROM)
#define ROM_0 stream_rom::S0
#define ROM_1 stream_rom::S1
#define ROM_2 stream_rom::S2
#define ROM_3 stream_rom::S3
#define ROM_4 stream_rom::S4
#define ROM_5 stream_rom::S5
#define ROM_8 stream_rom::S8
#else
#define ROM_0 nullptr
#define ROM_1 nullptr
#define ROM_2 nullptr
#define ROM_3 nullptr
#define ROM_4 nullptr
#define ROM_5 nullptr
#define ROM_8 nullptr
#endif
#define ROM_6 nullptr  // étages FRAME : poids en DDR
#define ROM_7 nullptr

void yolo_stream(hls::stream<axis_byte>& in, hls::stream<axis_byte>& out, const int8_t* wts,
                 const int32_t* bias, const int32_t* m0, StreamDesc d) {
#pragma HLS INTERFACE axis port=in
#pragma HLS INTERFACE axis port=out
#pragma HLS INTERFACE m_axi port=wts offset=slave bundle=gmem_w depth=15855552
#pragma HLS INTERFACE m_axi port=bias offset=slave bundle=gmem_p depth=3184
#pragma HLS INTERFACE m_axi port=m0 offset=slave bundle=gmem_p depth=3184
#pragma HLS INTERFACE s_axilite port=wts bundle=control
#pragma HLS INTERFACE s_axilite port=bias bundle=control
#pragma HLS INTERFACE s_axilite port=m0 bundle=control
#pragma HLS INTERFACE s_axilite port=d bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS DATAFLOW
#ifdef STREAM_ROM
  STREAM_ROM_PRAGMAS
#endif
  static hls::stream<int8_t> s_in("s_in"), s0("s0"), s1("s1"), s2("s2"), s3("s3"), s4("s4"),
      s5("s5"), s6("s6"), s7("s7"), s_head("s_head");
  // Une ligne de sortie du producteur (FIFO_DEPTH, stream_model.fifo_depths).
#pragma HLS STREAM variable=s_in depth=1248
#pragma HLS STREAM variable=s0 depth=3328
#pragma HLS STREAM variable=s1 depth=3328
#pragma HLS STREAM variable=s2 depth=3328
#pragma HLS STREAM variable=s3 depth=3328
#pragma HLS STREAM variable=s4 depth=3328
#pragma HLS STREAM variable=s5 depth=6656
#pragma HLS STREAM variable=s6 depth=13312
#pragma HLS STREAM variable=s7 depth=13312
#pragma HLS STREAM variable=s_head depth=1625
  from_axis(in, s_in);
  // K, C_in, C_out, H, W, pool k, pool s
  STAGE(0, 3, 3, 16, 416, 416, 2, 2)(s_in, s0, params(wts, bias, m0, d, 0, true) STAGE_ARGS(0));
  TAP(0, s0);
  STAGE(1, 3, 16, 32, 208, 208, 2, 2)(s0, s1, params(wts, bias, m0, d, 1, true) STAGE_ARGS(1));
  TAP(1, s1);
  STAGE(2, 3, 32, 64, 104, 104, 2, 2)(s1, s2, params(wts, bias, m0, d, 2, true) STAGE_ARGS(2));
  TAP(2, s2);
  STAGE(3, 3, 64, 128, 52, 52, 2, 2)(s2, s3, params(wts, bias, m0, d, 3, true) STAGE_ARGS(3));
  TAP(3, s3);
  STAGE(4, 3, 128, 256, 26, 26, 2, 2)(s3, s4, params(wts, bias, m0, d, 4, true) STAGE_ARGS(4));
  TAP(4, s4);
  STAGE(5, 3, 256, 512, 13, 13, 2, 1)(s4, s5, params(wts, bias, m0, d, 5, true) STAGE_ARGS(5));
  TAP(5, s5);
  STAGE(6, 3, 512, 1024, 13, 13, 0, 0)(s5, s6, params(wts, bias, m0, d, 6, true) STAGE_ARGS(6));
  TAP(6, s6);
  STAGE(7, 3, 1024, 1024, 13, 13, 0, 0)(s6, s7, params(wts, bias, m0, d, 7, true)
                                         STAGE_ARGS(7));
  TAP(7, s7);
  STAGE(8, 1, 1024, 125, 13, 13, 0, 0)(s7, s_head, params(wts, bias, m0, d, 8, false)
                                        STAGE_ARGS(8));
  TAP(8, s_head);
  to_axis(s_head, out);
}
