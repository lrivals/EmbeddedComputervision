#include "accel_driver.hpp"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

#include "regmap.hpp"

namespace driver {

static int64_t align64(int64_t x) { return (x + ARENA_ALIGN - 1) / ARENA_ALIGN * ARENA_ALIGN; }

// Le noyau n'a aucune garde d'adresse : toute lecture ou écriture d'un `LayerDesc` doit
// tomber dans sa zone, sinon il corromprait la DDR voisine.
static void check_bounds(const ConvCall& c, int64_t arena, int64_t weights, int64_t params) {
  const accel::LayerDesc& d = c.desc;
  auto in = [&](int64_t off, int64_t n, int64_t size, const char* what) {
    if (off < 0 || n < 0 || off + n > size)
      throw std::runtime_error("conv " + std::to_string(c.layer) + " : " + what +
                               " hors de sa zone");
  };
  in(d.seg0_off, int64_t(d.seg0_c) * d.seg0_h * d.seg0_w, arena, "segment 0");
  if (d.nseg > 1) in(d.seg1_off, int64_t(d.seg1_c) * d.seg1_h * d.seg1_w, arena, "segment 1");
  in(d.out_off, int64_t(d.cout) * d.pool_h() * d.pool_w(), arena, "sortie");
  if (d.prepool_off >= 0)
    in(d.prepool_off, int64_t(d.cout) * d.out_h() * d.out_w(), arena, "carte avant pooling");
  in(d.w_off, int64_t(d.cout) * d.cin * d.k * d.k, weights, "poids");
  in(d.b_off, d.cout, params, "biais");
  in(d.m0_off, d.cout, params, "M0");
}

Accelerator::Accelerator(Device& dev, const golden::Model& m)
    : dev_(dev), m_(m), prog_(build(m)) {
  const int64_t w_at = align64(prog_.arena_size);
  const int64_t p_at = align64(w_at + int64_t(m.weights.size()));
  const int64_t params_bytes = int64_t(prog_.params.size() * sizeof(int32_t));
  mem_ = dev_.alloc(size_t(p_at + params_bytes));
  if (mem_.phys % ARENA_ALIGN) throw std::runtime_error("tampon contigu non aligné à 64");
  for (const ConvCall& c : prog_.calls)
    check_bounds(c, prog_.arena_size, int64_t(m.weights.size()), int64_t(prog_.params.size()));
  std::memset(mem_.virt, 0, size_t(w_at));
  std::memcpy(mem_.virt + w_at, m.weights.data(), m.weights.size());
  std::memcpy(mem_.virt + p_at, prog_.params.data(), size_t(params_bytes));
  w_phys_ = mem_.phys + uint64_t(w_at);
  p_phys_ = mem_.phys + uint64_t(p_at);
  dev_.sync_for_device(mem_);
}

void Accelerator::load_input(const int8_t* input) {
  std::memset(mem_.virt, 0, size_t(prog_.arena_size));
  std::memcpy(arena() + prog_.input_off, input, size_t(m_.in_c) * m_.in_h * m_.in_w);
  dev_.sync_for_device(mem_);
}

double Accelerator::run_layer(const ConvCall& c) {
  const auto t0 = std::chrono::steady_clock::now();
  if (!(dev_.read32(regmap::CTRL) & regmap::AP_IDLE))
    throw std::runtime_error("yolo_conv occupé avant la conv " + std::to_string(c.layer));
  auto ptr = [&](uint32_t off, uint64_t phys) {
    dev_.write32(off, uint32_t(phys));
    dev_.write32(off + 4, uint32_t(phys >> 32));
  };
  ptr(regmap::ACT_IN, mem_.phys);  // même arène sur les deux bundles (conventions.md)
  ptr(regmap::ACT_OUT, mem_.phys);
  ptr(regmap::WTS, w_phys_);
  ptr(regmap::PRM, p_phys_);
  uint32_t w[regmap::D_WORDS];
  regmap::desc_to_words(c.desc, w);
  for (int i = 0; i < regmap::D_WORDS; ++i) dev_.write32(regmap::D + 4 * uint32_t(i), w[i]);
  dev_.write32(regmap::CTRL, regmap::AP_START);
  dev_.wait_done();
  dev_.sync_for_cpu(mem_);
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void Accelerator::run(const int8_t* input,
                      const std::function<void(const ConvCall&, double)>& after) {
  load_input(input);
  for (const ConvCall& c : prog_.calls) {
    const double s = run_layer(c);
    if (after) after(c, s);
  }
}

}  // namespace driver
