// Scalar operators of the golden model on files (T28.1, T26.2).
//
//   golden_ops requant <ops.bin> <out.bin>   → output stage (requantize, leaky / SiLU, clip_q) of
//                                              every accumulator; file format in
//                                              python/yolo/quant/bounds.py (`write_ops`)
//   golden_ops luts <model_dir>              → per head, the σ, e^t and softmax tables as the
//                                              hardware post-processing indexes them, one
//                                              line per input: head t sig[t] exp[t] smexp[d]
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "golden/conv.hpp"
#include "golden/hw_postproc.hpp"
#include "golden/model.hpp"

using namespace golden;

static int usage() {
  std::cerr << "usage : golden_ops requant <ops.bin> <out.bin>\n"
               "        golden_ops luts <model_dir>\n";
  return 2;
}

static int requant(const std::string& in, const std::string& out) {
  std::ifstream f(in, std::ios::binary);
  int32_t head[3];
  if (!f.read(reinterpret_cast<char*>(head), sizeof head)) return usage();
  const int nb = head[0], B = head[1];
  int8_t lut[256] = {};
  if (head[2] == 64) f.read(reinterpret_cast<char*>(lut), sizeof lut);  // SiLU (T26.2)
  std::vector<int8_t> y(size_t(nb) * B);
  std::vector<int32_t> blk(size_t(4 + B));
  for (int b = 0; b < nb; ++b) {
    if (!f.read(reinterpret_cast<char*>(blk.data()), std::streamsize(blk.size() * 4))) {
      std::cerr << in << " : truncated\n";
      return 1;
    }
    const int32_t m0 = blk[0], n = blk[1], qmax = blk[2], act = blk[3];
    for (int k = 0; k < B; ++k) {
      int32_t v = requantize(blk[size_t(4 + k)], m0, n);
      if (act == 2) {
        y[size_t(b) * B + k] = silu_lookup(v, lut);
        continue;
      }
      if (act) v = leaky_int(v);
      y[size_t(b) * B + k] = clip_q(v, qmax);
    }
  }
  std::ofstream(out, std::ios::binary)
      .write(reinterpret_cast<const char*>(y.data()), std::streamsize(y.size()));
  return 0;
}

static int luts(const std::string& dir) {
  const Model m = Model::load(dir);
  for (int id : m.heads()) {
    const Layer& l = m.layers[size_t(id)];
    const uint32_t* sig = m.head_lut(l, 0);
    const uint32_t* ex = m.head_lut(l, 1);
    const uint32_t* sm = m.head_lut(l, 2);
    for (int t = -128; t < 128; ++t)
      std::printf("%d %d %u %u %u\n", id, t, sig[t + hwpp::LUT_OFFSET], ex[t + hwpp::LUT_OFFSET],
                  sm[(t - 127) + hwpp::SOFTMAX_OFFSET]);  // d = t − 127 ∈ [−255, 0]
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  const std::string cmd = argv[1];
  if (cmd == "requant" && argc == 4) return requant(argv[2], argv[3]);
  if (cmd == "luts" && argc == 3) return luts(argv[2]);
  return usage();
}
