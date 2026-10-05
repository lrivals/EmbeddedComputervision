// Top streaming de Tiny-YOLOv2 (T9.4.2). Voir yolo_stream.hpp.
#include "yolo_stream.hpp"

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

}  // namespace stream

using namespace stream;

void yolo_stream(hls::stream<int8_t>& in, hls::stream<int8_t>& out, const int8_t* wts,
                 const int32_t* bias, const int32_t* m0, StreamDesc d) {
#pragma HLS INTERFACE axis port=in
#pragma HLS INTERFACE axis port=out
#pragma HLS INTERFACE m_axi port=wts offset=slave bundle=gmem_w depth=15855552
#pragma HLS INTERFACE m_axi port=bias offset=slave bundle=gmem_p depth=3184
#pragma HLS INTERFACE m_axi port=m0 offset=slave bundle=gmem_p depth=3184
#pragma HLS INTERFACE s_axilite port=d bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS DATAFLOW
  static hls::stream<int8_t> s0("s0"), s1("s1"), s2("s2"), s3("s3"), s4("s4"), s5("s5"),
      s6("s6"), s7("s7");
  // K, C_in, C_out, H, W, PE, SIMD, pool k, pool s, carte entière (poids en DDR)
  conv_stage<3, 3, 16, 416, 416, 8, 3, 2, 2, false>(in, s0, params(wts, bias, m0, d, 0, true)
                                                    STAGE_ARGS(0));
  TAP(0, s0);
  conv_stage<3, 16, 32, 208, 208, 4, 16, 2, 2, false>(s0, s1, params(wts, bias, m0, d, 1, true)
                                                      STAGE_ARGS(1));
  TAP(1, s1);
  conv_stage<3, 32, 64, 104, 104, 2, 32, 2, 2, false>(s1, s2, params(wts, bias, m0, d, 2, true)
                                                      STAGE_ARGS(2));
  TAP(2, s2);
  conv_stage<3, 64, 128, 52, 52, 1, 64, 2, 2, false>(s2, s3, params(wts, bias, m0, d, 3, true)
                                                     STAGE_ARGS(3));
  TAP(3, s3);
  conv_stage<3, 128, 256, 26, 26, 1, 64, 2, 2, false>(s3, s4, params(wts, bias, m0, d, 4, true)
                                                      STAGE_ARGS(4));
  TAP(4, s4);
  conv_stage<3, 256, 512, 13, 13, 1, 64, 2, 1, false>(s4, s5, params(wts, bias, m0, d, 5, true)
                                                      STAGE_ARGS(5));
  TAP(5, s5);
  conv_stage<3, 512, 1024, 13, 13, 1, 256, 0, 0, true>(s5, s6,
                                                       params(wts, bias, m0, d, 6, true)
                                                       STAGE_ARGS(6));
  TAP(6, s6);
  conv_stage<3, 1024, 1024, 13, 13, 1, 256, 0, 0, true>(s6, s7,
                                                        params(wts, bias, m0, d, 7, true)
                                                        STAGE_ARGS(7));
  TAP(7, s7);
  conv_stage<1, 1024, 125, 13, 13, 1, 32, 0, 0, false>(s7, out,
                                                       params(wts, bias, m0, d, 8, false)
                                                       STAGE_ARGS(8));
  TAP(8, out);
}
