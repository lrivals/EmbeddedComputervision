#include "stream_driver.hpp"

#include <chrono>
#include <cstring>
#include <stdexcept>

#include "stream_regmap.hpp"

namespace driver {

static int64_t align64(int64_t x) { return (x + 63) / 64 * 64; }

stream::StreamDesc stream_desc(const golden::Model& m) {
  stream::StreamDesc d{};
  for (int k = 0; k < stream::N_STAGES; ++k) {
    const size_t id = size_t(stream::STAGE_LAYER[k]);
    if (id >= m.layers.size() || m.layers[id].type != golden::LayerType::Conv)
      throw std::runtime_error("streaming : Tiny-YOLOv2 VOC attendu (étage sans conv)");
    const golden::Layer& l = m.layers[id];
    d.w_off[k] = int32_t(l.w_offset);
    d.b_off[k] = int32_t(l.b_offset / 4);
    d.m0_off[k] = int32_t(l.m0_offset / 4);
    d.shift[k] = l.shift;
    d.qmax[k] = l.qmax;
  }
  return d;
}

StreamAccelerator::StreamAccelerator(Device& dev, const golden::Model& m, int n_inputs)
    : dev_(dev), m_(m), n_inputs_(n_inputs) {
  if (!dev.has_stream()) dev.stream_write32(0, 0);  // lève « yolo_stream absent »
  if (m.heads().size() != 1 || int64_t(m.in_c) * m.in_h * m.in_w != stream::IN_VALUES)
    throw std::runtime_error("streaming : Tiny-YOLOv2 VOC (416 × 416, une tête) attendu");
  head_id_ = m.heads()[0];
  const golden::Layer& h = m.layers[size_t(head_id_)];
  if (int64_t(h.out_c) * h.out_h * h.out_w != stream::OUT_VALUES)
    throw std::runtime_error("streaming : tête (125, 13, 13) attendue");
  in_stride_ = align64(stream::IN_VALUES);
  out_at_ = in_stride_ * n_inputs;
  w_at_ = align64(out_at_ + stream::OUT_VALUES);
  b_at_ = align64(w_at_ + int64_t(m.weights.size()));
  m0_at_ = align64(b_at_ + 4 * int64_t(m.bias.size()));
  mem_ = dev.alloc(size_t(align64(m0_at_ + 4 * int64_t(m.m0.size()))));
  std::memcpy(mem_.virt + w_at_, m.weights.data(), m.weights.size());
  std::memcpy(mem_.virt + b_at_, m.bias.data(), 4 * m.bias.size());
  std::memcpy(mem_.virt + m0_at_, m.m0.data(), 4 * m.m0.size());
  dev.sync_for_device(mem_, size_t(w_at_), mem_.size - size_t(w_at_));

  // Registres constants : pointeurs des blobs et descripteur.
  auto ptr = [&](uint32_t off, int64_t at) {
    const uint64_t a = mem_.phys + uint64_t(at);
    dev.stream_write32(off, uint32_t(a));
    dev.stream_write32(off + 4, uint32_t(a >> 32));
  };
  ptr(stream_regmap::WTS, w_at_);
  ptr(stream_regmap::BIAS, b_at_);
  ptr(stream_regmap::M0, m0_at_);
  uint32_t w[stream_regmap::D_WORDS];
  stream_regmap::desc_to_words(stream_desc(m), w);
  for (int i = 0; i < stream_regmap::D_WORDS; ++i)
    dev.stream_write32(stream_regmap::D + 4 * uint32_t(i), w[i]);
  using namespace dma_regmap;
  dev.dma_write32(MM2S_DMACR, CR_RS);
  dev.dma_write32(S2MM_DMACR, CR_RS | CR_IOC_IRQ_EN | CR_ERR_IRQ_EN);
  head_.resize(stream::OUT_VALUES);
}

void StreamAccelerator::load_input(const int8_t* chw, int slot) {
  if (slot < 0 || slot >= n_inputs_) throw std::out_of_range("streaming : emplacement");
  const int C = m_.in_c, H = m_.in_h, W = m_.in_w;
  int8_t* dst = reinterpret_cast<int8_t*>(mem_.virt + slot * in_stride_);
  for (int r = 0; r < H; ++r)
    for (int c = 0; c < W; ++c)
      for (int ch = 0; ch < C; ++ch) *dst++ = chw[(size_t(ch) * H + r) * W + c];
  dev_.sync_for_device(mem_, size_t(slot * in_stride_), size_t(stream::IN_VALUES));
}

double StreamAccelerator::run(int slot) {
  using namespace dma_regmap;
  if (slot < 0 || slot >= n_inputs_) throw std::out_of_range("streaming : emplacement");
  const auto t0 = std::chrono::steady_clock::now();
  dev_.stream_write32(stream_regmap::CTRL, stream_regmap::AP_START);
  const uint64_t da = mem_.phys + uint64_t(out_at_);
  const uint64_t sa = mem_.phys + uint64_t(slot * in_stride_);
  dev_.dma_write32(S2MM_DA, uint32_t(da));
  dev_.dma_write32(S2MM_DA_MSB, uint32_t(da >> 32));
  dev_.dma_write32(S2MM_LENGTH, uint32_t(stream::OUT_VALUES));
  dev_.dma_write32(MM2S_SA, uint32_t(sa));
  dev_.dma_write32(MM2S_SA_MSB, uint32_t(sa >> 32));
  dev_.dma_write32(MM2S_LENGTH, uint32_t(stream::IN_VALUES));
  dev_.dma_wait_done();
  const uint32_t got = dev_.dma_read32(S2MM_LENGTH);
  if (got != uint32_t(stream::OUT_VALUES))
    throw std::runtime_error("streaming : tête de " + std::to_string(got) + " octets");
  dev_.sync_for_cpu(mem_, size_t(out_at_), size_t(stream::OUT_VALUES));
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  // HWC → CHW, l'ordre des têtes du golden et du post-traitement.
  const golden::Layer& h = m_.layers[size_t(head_id_)];
  const int8_t* src = reinterpret_cast<const int8_t*>(mem_.virt + out_at_);
  for (int r = 0; r < h.out_h; ++r)
    for (int c = 0; c < h.out_w; ++c)
      for (int ch = 0; ch < h.out_c; ++ch)
        head_[(size_t(ch) * h.out_h + r) * h.out_w + c] = *src++;
  return s;
}

}  // namespace driver
