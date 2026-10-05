#include "accel_driver.hpp"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

#include "post_regmap.hpp"
#include "post_table.hpp"
#include "regmap.hpp"
#include "weight_layout.hpp"

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
  in(d.w_off, accel::w_layer_bytes(d), weights, "poids");
  in(d.b_off, d.cout, params, "biais");
  in(d.m0_off, d.cout, params, "M0");
}

Accelerator::Accelerator(Device& dev, const golden::Model& m, int n_arenas)
    : dev_(dev), m_(m), prog_(build(m)), n_arenas_(n_arenas) {
  if (n_arenas < 1) throw std::runtime_error("au moins une arène");
  arena_stride_ = align64(prog_.arena_size);
  const int64_t w_at = arena_stride_ * n_arenas;
  const int64_t p_at = align64(w_at + int64_t(prog_.weights.size()));
  const int64_t params_bytes = int64_t(prog_.params.size() * sizeof(int32_t));
  const std::vector<int32_t> descs = desc_table(prog_);
  const int64_t d_at = align64(p_at + params_bytes);
  int64_t end = d_at + int64_t(descs.size() * sizeof(int32_t));
  if (dev_.has_post()) {
    for (int id : m.heads()) {
      const View& v = prog_.views[size_t(id)];
      if (v.nseg != 1 || v.seg[0].up != 0) throw std::runtime_error("tête non contiguë");
      head_off_.push_back(v.seg[0].off);
    }
    const size_t tab_words = post_table(m, head_off_, 0.25, 0.45).words.size();
    tab_at_ = align64(end);
    res_at_ = align64(tab_at_ + int64_t(tab_words * 4));
    end = res_at_ + 4 * (2 + accel::POST_BOX_WORDS * accel::POST_CAP);
  }
  mem_ = dev_.alloc(size_t(end));
  if (mem_.phys % ARENA_ALIGN) throw std::runtime_error("tampon contigu non aligné à 64");
  for (const ConvCall& c : prog_.calls)
    check_bounds(c, prog_.arena_size, int64_t(prog_.weights.size()),
                 int64_t(prog_.params.size()));
  std::memset(mem_.virt, 0, size_t(w_at));
  std::memcpy(mem_.virt + w_at, prog_.weights.data(), prog_.weights.size());
  std::memcpy(mem_.virt + p_at, prog_.params.data(), size_t(params_bytes));
  std::memcpy(mem_.virt + d_at, descs.data(), descs.size() * sizeof(int32_t));
  w_phys_ = mem_.phys + uint64_t(w_at);
  p_phys_ = mem_.phys + uint64_t(p_at);
  desc_phys_ = mem_.phys + uint64_t(d_at);
  dev_.sync_for_device(mem_);
}

void Accelerator::load_input(const int8_t* input, int slot) {
  std::memset(arena(slot), 0, size_t(prog_.arena_size));
  std::memcpy(arena(slot) + prog_.input_off, input, size_t(m_.in_c) * m_.in_h * m_.in_w);
  dev_.sync_for_device(mem_, size_t(slot * arena_stride_), size_t(prog_.arena_size));
}

void Accelerator::set_ptrs(int slot) {
  auto ptr = [&](uint32_t off, uint64_t phys) {
    dev_.write32(off, uint32_t(phys));
    dev_.write32(off + 4, uint32_t(phys >> 32));
  };
  ptr(regmap::ACT_IN, arena_phys(slot));  // même arène sur les deux bundles (conventions.md)
  ptr(regmap::ACT_OUT, arena_phys(slot));
  ptr(regmap::WTS, w_phys_);
  ptr(regmap::PRM, p_phys_);
  ptr(regmap::DESCS, desc_phys_);
}

void Accelerator::start_and_wait(int slot) {
  dev_.write32(regmap::CTRL, regmap::AP_START);
  dev_.wait_done();
  dev_.sync_for_cpu(mem_, size_t(slot * arena_stride_), size_t(prog_.arena_size));
}

double Accelerator::run_layer(const ConvCall& c, int slot) {
  const auto t0 = std::chrono::steady_clock::now();
  if (!(dev_.read32(regmap::CTRL) & regmap::AP_IDLE))
    throw std::runtime_error("yolo_conv occupé avant la conv " + std::to_string(c.layer));
  set_ptrs(slot);
  uint32_t w[regmap::D_WORDS];
  regmap::desc_to_words(c.desc, w);
  for (int i = 0; i < regmap::D_WORDS; ++i) dev_.write32(regmap::D + 4 * uint32_t(i), w[i]);
  dev_.write32(regmap::N_CALLS, 0);
  start_and_wait(slot);
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

double Accelerator::run_all(int slot) {
  const auto t0 = std::chrono::steady_clock::now();
  if (!(dev_.read32(regmap::CTRL) & regmap::AP_IDLE))
    throw std::runtime_error("yolo_conv occupé avant la passe");
  set_ptrs(slot);
  dev_.write32(regmap::N_CALLS, uint32_t(prog_.calls.size()));
  start_and_wait(slot);
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

std::vector<hwpp::Box> Accelerator::run_post(double conf, double iou, int* overflow,
                                             double* seconds, int slot) {
  const auto t0 = std::chrono::steady_clock::now();
  if (!dev_.has_post()) throw std::runtime_error("yolo_post absent du périphérique");
  if (conf != post_conf_ || iou != post_iou_) {  // table écrite une fois par réglage
    const PostTable t = post_table(m_, head_off_, conf, iou);
    std::memcpy(mem_.virt + tab_at_, t.words.data(), t.words.size() * 4);
    dev_.sync_for_device(mem_, size_t(tab_at_), t.words.size() * 4);
    post_desc_ = t.desc;
    post_conf_ = conf;
    post_iou_ = iou;
  }
  if (!(dev_.post_read32(post_regmap::CTRL) & post_regmap::AP_IDLE))
    throw std::runtime_error("yolo_post occupé");
  auto ptr = [&](uint32_t off, uint64_t phys) {
    dev_.post_write32(off, uint32_t(phys));
    dev_.post_write32(off + 4, uint32_t(phys >> 32));
  };
  ptr(post_regmap::ACT, arena_phys(slot));  // têtes lues dans l'arène (T10.7)
  ptr(post_regmap::TAB, mem_.phys + uint64_t(tab_at_));
  ptr(post_regmap::RES, mem_.phys + uint64_t(res_at_));
  uint32_t w[post_regmap::D_WORDS];
  post_regmap::desc_to_words(post_desc_, w);
  for (int i = 0; i < post_regmap::D_WORDS; ++i)
    dev_.post_write32(post_regmap::D + 4 * uint32_t(i), w[i]);
  dev_.post_write32(post_regmap::CTRL, post_regmap::AP_START);
  dev_.post_wait_done();
  dev_.sync_for_cpu(mem_, size_t(res_at_),
                    size_t(4 * (2 + accel::POST_BOX_WORDS * accel::POST_CAP)));
  const auto out = post_boxes(reinterpret_cast<const int32_t*>(mem_.virt + res_at_), overflow);
  if (seconds)
    *seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return out;
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
