// Backend `sim` (PC) : banc de registres AXI-Lite émulé devant le noyau C-sim `yolo_conv`.
//
// Les tampons reçoivent des adresses physiques fictives (au-delà de 4 Go, pour exercer les
// mots de poids fort des pointeurs 64 bits). ap_start → relecture des registres, traduction
// physique → virtuelle, appel du noyau, ap_done (effacé à la lecture, comme ap_ctrl_hs).
#include <map>
#include <stdexcept>
#include <vector>

#include "accel.hpp"
#include "device.hpp"
#include "regmap.hpp"

namespace driver {
namespace {

class SimDevice : public Device {
 public:
  SimDevice() { regs_[regmap::CTRL / 4] = regmap::AP_IDLE; }

  const char* name() const override { return "sim"; }

  Buffer alloc(size_t size) override {
    mem_.emplace_back(size);
    Buffer b{mem_.back().data(), next_phys_, size};
    bufs_[b.phys] = b;
    next_phys_ += (uint64_t(size) + PAGE - 1) / PAGE * PAGE + PAGE;  // page de garde
    return b;
  }

  void write32(uint32_t off, uint32_t v) override {
    check(off);
    if (off == regmap::CTRL) {
      if (v & regmap::AP_START) start();
      return;
    }
    if (off == regmap::ISR) {
      regs_[off / 4] &= ~v;  // basculement par écriture de 1
      return;
    }
    regs_[off / 4] = v;
  }

  uint32_t read32(uint32_t off) override {
    check(off);
    const uint32_t v = regs_[off / 4];
    if (off == regmap::CTRL) regs_[off / 4] &= ~regmap::AP_DONE;  // effacé à la lecture
    return v;
  }

  void wait_done() override {
    if (!(read32(regmap::CTRL) & regmap::AP_DONE))
      throw std::runtime_error("sim : ap_done absent après ap_start");
  }

 private:
  static constexpr uint64_t PAGE = 4096;

  static void check(uint32_t off) {
    if (off % 4 || off >= regmap::SPAN) throw std::out_of_range("sim : registre hors fenêtre");
  }

  uint64_t reg64(uint32_t off) const {
    return uint64_t(regs_[off / 4]) | uint64_t(regs_[off / 4 + 1]) << 32;
  }

  // Adresse physique → pointeur dans le tampon qui la contient.
  uint8_t* virt(uint64_t phys) const {
    auto it = bufs_.upper_bound(phys);
    if (it == bufs_.begin()) throw std::runtime_error("sim : adresse physique non allouée");
    --it;
    const Buffer& b = it->second;
    if (phys >= b.phys + b.size) throw std::runtime_error("sim : adresse physique hors tampon");
    return b.virt + (phys - b.phys);
  }

  void start() {
    regs_[regmap::CTRL / 4] &= ~(regmap::AP_IDLE | regmap::AP_DONE);
    const accel::LayerDesc d = regmap::words_to_desc(&regs_[regmap::D / 4]);
    yolo_conv(reinterpret_cast<const int8_t*>(virt(reg64(regmap::ACT_IN))),
              reinterpret_cast<int8_t*>(virt(reg64(regmap::ACT_OUT))),
              reinterpret_cast<const int8_t*>(virt(reg64(regmap::WTS))),
              reinterpret_cast<const int32_t*>(virt(reg64(regmap::PRM))), d);
    regs_[regmap::CTRL / 4] |= regmap::AP_DONE | regmap::AP_IDLE | regmap::AP_READY;
    if ((regs_[regmap::IER / 4] & 1)) regs_[regmap::ISR / 4] |= 1;
  }

  uint32_t regs_[regmap::SPAN / 4] = {};
  std::vector<std::vector<uint8_t>> mem_;
  std::map<uint64_t, Buffer> bufs_;  // par adresse physique de début
  uint64_t next_phys_ = 0x8'0000'0000ull;
};

}  // namespace

std::unique_ptr<Device> make_sim_device() { return std::make_unique<SimDevice>(); }

}  // namespace driver
