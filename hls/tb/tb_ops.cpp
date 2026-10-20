// Output stage of the HLS kernel on swept accumulators (T28.1): `store_tile` (bias,
// requantization, leaky, saturation, write to DDR) on synthetic tiles, compared with the golden
// scalar operators — or, with --ops, run on an operator file and written out for the Python
// comparison (python/tests/test_exhaustive_ops.py; format in python/yolo/quant/bounds.py).
//
//   tb_ops                          self-check: random (M0, n) and the rounding boundaries
//                                   (linear, leaky; SiLU with python/tests/test_exhaustive_ops.py)
//   tb_ops --ops <in.bin> --out <out.bin>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "output_stage.hpp"

using namespace accel;

namespace {

// One call of store_tile: up to TM blocks of tr·tc accumulators sharing (n, qmax, leaky), one
// output channel per block (its own M0), no bias, no pooling.
struct Group {
  int n = 0, qmax = 0, leaky = 0;  // leaky: activation code (2: SiLU, T26.2)
  const int8_t* lut = nullptr;
  std::vector<int32_t> m0;
  std::vector<const int32_t*> acc;
};

void run_group(const Group& g, int tr, int tc, int8_t* out) {
  static acc_t buf[TM][TRB][TCB];
  const int nb = int(g.m0.size());
  LayerDesc d{};
  d.k = 1;
  d.pad = 0;
  d.cin = 1;
  d.cout = nb;
  d.h = tr;
  d.w = tc;
  d.shift = g.n;
  d.leaky = g.leaky;
  d.pool_k = d.pool_s = 0;
  d.prepool_off = -1;
  d.out_off = 0;
  d.b_off = 0;
  d.m0_off = TM;
  d.qmax = g.qmax;
  d.tr = tr;
  d.tc = tc;
  std::vector<int32_t> prm(2 * TM + 64, 0);
  for (int b = 0; b < nb; ++b) prm[size_t(TM + b)] = g.m0[size_t(b)];
  if (g.lut) std::memcpy(&prm[size_t(TM + nb)], g.lut, 256);  // after the M0s (SiLU table)
  for (int b = 0; b < nb; ++b)
    for (int i = 0; i < tr; ++i)
      for (int j = 0; j < tc; ++j) buf[b][i][j] = g.acc[size_t(b)][i * tc + j];
  TileInfo t{};
  t.tr_n = t.np_r = tr;
  t.tc_n = t.np_c = tc;
  t.tm_n = nb;
  const int bytes = nb * tr * tc;
  std::vector<word_t> mem(size_t(bytes / WORD + 2), 0);
  store_tile(mem.data(), prm.data(), d, t, buf);
  std::memcpy(out, mem.data(), size_t(bytes));
}

int file_mode(const std::string& in, const std::string& out) {
  std::ifstream f(in, std::ios::binary);
  int32_t head[3];
  if (!f.read(reinterpret_cast<char*>(head), sizeof head)) return 2;
  const int nb = head[0], B = head[1];
  static int8_t lut[256];
  if (head[2] == 64) f.read(reinterpret_cast<char*>(lut), sizeof lut);
  int tr = 1;
  while ((tr + 1) * (tr + 1) <= B) ++tr;
  if (tr * tr != B || tr > TRB) {
    std::fprintf(stderr, "B = %d: not a square tile ≤ %d²\n", B, TRB);
    return 2;
  }
  std::vector<int32_t> data(size_t(nb) * (4 + B));
  f.read(reinterpret_cast<char*>(data.data()), std::streamsize(data.size() * 4));
  std::vector<int8_t> y(size_t(nb) * B);
  Group g;
  int first = 0;
  auto flush = [&](int end) {
    if (!g.m0.empty()) run_group(g, tr, tr, y.data() + size_t(first) * B);
    g = Group{};
    first = end;
  };
  for (int b = 0; b < nb; ++b) {
    const int32_t* blk = data.data() + size_t(b) * (4 + B);
    if (!g.m0.empty() &&
        (blk[1] != g.n || blk[2] != g.qmax || blk[3] != g.leaky || int(g.m0.size()) == TM))
      flush(b);
    g.n = blk[1];
    g.qmax = blk[2];
    g.leaky = blk[3];
    g.lut = blk[3] == 2 ? lut : nullptr;
    g.m0.push_back(blk[0]);
    g.acc.push_back(blk + 4);
  }
  flush(nb);
  std::ofstream(out, std::ios::binary)
      .write(reinterpret_cast<const char*>(y.data()), std::streamsize(y.size()));
  std::printf("%d blocks of %d accumulators\n", nb, B);
  return 0;
}

int self_check() {
  std::mt19937 rng(28);
  std::uniform_int_distribution<int32_t> any(INT32_MIN, INT32_MAX);
  const int tr = TR, tc = TC, B = tr * tc;
  long long checked = 0, bad = 0;
  for (int it = 0; it < 4000; ++it) {
    Group g;
    g.n = 1 + int(rng() % 31);
    g.qmax = (it % 3 == 0) ? 7 : 127;  // 4-bit outputs (T9.3) too
    g.leaky = it % 2;
    std::vector<std::vector<int32_t>> acc(TM, std::vector<int32_t>(size_t(B)));
    for (int b = 0; b < TM; ++b) {
      const int32_t m0 = int32_t(rng() & 0x7fffffff);
      g.m0.push_back(m0);
      // Around the boundaries k·2ⁿ − 2ⁿ⁻¹ divided by M0, plus the int32 extremes.
      for (int k = 0; k < B; ++k) {
        int64_t v;
        if (k < 4) {
          const int64_t ext[4] = {INT32_MIN, INT32_MAX, 0, -1};
          v = ext[k];
        } else if (k % 2 && m0 > 0) {
          const int64_t q = int64_t(rng() % 2600) - 1300;
          v = ((q << g.n) - (int64_t(1) << (g.n - 1))) / m0 + int64_t(rng() % 3) - 1;
        } else {
          v = any(rng);
        }
        if (v < INT32_MIN) v = INT32_MIN;
        if (v > INT32_MAX) v = INT32_MAX;
        acc[size_t(b)][size_t(k)] = int32_t(v);
      }
      g.acc.push_back(acc[size_t(b)].data());
    }
    std::vector<int8_t> y(size_t(TM) * B);
    run_group(g, tr, tc, y.data());
    for (int b = 0; b < TM; ++b)
      for (int k = 0; k < B; ++k) {
        int32_t v = golden::requantize(acc[size_t(b)][size_t(k)], g.m0[size_t(b)], g.n);
        if (g.leaky) v = golden::leaky_int(v);
        const int8_t want = golden::clip_q(v, g.qmax);
        ++checked;
        if (y[size_t(b) * B + k] != want && bad++ < 5)
          std::printf("acc %d M0 %d n %d: %d instead of %d\n", acc[size_t(b)][size_t(k)],
                      g.m0[size_t(b)], g.n, int(y[size_t(b) * B + k]), int(want));
      }
  }
  std::printf("store_tile == golden on %lld accumulators, %lld differences\n", checked, bad);
  return bad == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  std::string in, out;
  for (int k = 1; k + 1 < argc; k += 2) {
    if (!std::strcmp(argv[k], "--ops")) in = argv[k + 1];
    else if (!std::strcmp(argv[k], "--out")) out = argv[k + 1];
  }
  if (!in.empty() && !out.empty()) return file_mode(in, out);
  return self_check();
}
