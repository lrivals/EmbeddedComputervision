// Backend `uio` (KV260, Ubuntu Kria) : registres par /dev/uioN, tampon contigu par u-dma-buf.
//
// - Le dtbo (hw/boards/kv260/pl.dtsi) déclare `s_axi_control` en `generic-uio` avec
//   l'interruption du noyau, et un nœud `u-dma-buf` (CMA) assez grand pour arène + poids +
//   paramètres.
// - Les ports HP ne sont pas cohérents avec les caches de l'ARM : le u-dma-buf est ouvert en
//   O_SYNC (non caché), ce qui rend `sync_for_*` inutiles au prix d'accès CPU plus lents.
// - Interruption UIO : écrire 1 dans le fd la démasque, `read` bloque jusqu'à la suivante ;
//   l'ISR du noyau HLS est acquittée par écriture de 1.
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

#include "device.hpp"
#include "regmap.hpp"

namespace driver {
namespace {

std::runtime_error sys_error(const std::string& what) {
  return std::runtime_error(what + " : " + std::strerror(errno));
}

template <class T>
T read_sysfs(const std::string& path) {
  std::ifstream f(path);
  T v{};
  if (!(f >> v)) throw std::runtime_error("lecture impossible : " + path);
  return v;
}

class UioDevice : public Device {
 public:
  explicit UioDevice(const DeviceOptions& o) : irq_(o.irq), udmabuf_(o.udmabuf) {
    uio_fd_ = ::open(o.uio.c_str(), O_RDWR | O_SYNC);
    if (uio_fd_ < 0) throw sys_error("ouverture de " + o.uio);
    void* p = ::mmap(nullptr, regmap::SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, uio_fd_, 0);
    if (p == MAP_FAILED) throw sys_error("mmap des registres");
    regs_ = static_cast<volatile uint32_t*>(p);
    if (!(read32(regmap::CTRL) & regmap::AP_IDLE))
      throw std::runtime_error("yolo_conv n'est pas au repos (bitstream chargé ?)");
    write32(regmap::GIE, irq_ ? 1 : 0);
    write32(regmap::IER, irq_ ? 1 : 0);
  }

  ~UioDevice() override {
    if (buf_.virt) ::munmap(buf_.virt, buf_.size);
    if (dma_fd_ >= 0) ::close(dma_fd_);
    if (regs_) ::munmap(const_cast<uint32_t*>(regs_), regmap::SPAN);
    if (uio_fd_ >= 0) ::close(uio_fd_);
  }

  const char* name() const override { return "uio"; }

  Buffer alloc(size_t size) override {
    if (buf_.virt) throw std::runtime_error("uio : un seul tampon (le u-dma-buf entier)");
    const std::string sys = "/sys/class/u-dma-buf/" + udmabuf_ + "/";
    const size_t avail = read_sysfs<size_t>(sys + "size");
    if (size > avail)
      throw std::runtime_error("u-dma-buf trop petit : " + std::to_string(size) + " > " +
                               std::to_string(avail) + " octets (pl.dtsi)");
    std::ifstream f(sys + "phys_addr");
    std::string hex;
    if (!(f >> hex)) throw std::runtime_error("lecture impossible : " + sys + "phys_addr");
    const uint64_t phys = std::stoull(hex, nullptr, 16);
    const std::string dev = "/dev/" + udmabuf_;
    dma_fd_ = ::open(dev.c_str(), O_RDWR | O_SYNC);
    if (dma_fd_ < 0) throw sys_error("ouverture de " + dev);
    void* p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, dma_fd_, 0);
    if (p == MAP_FAILED) throw sys_error("mmap de " + dev);
    buf_ = Buffer{static_cast<uint8_t*>(p), phys, size};
    return buf_;
  }

  void write32(uint32_t off, uint32_t v) override { regs_[off / 4] = v; }
  uint32_t read32(uint32_t off) override { return regs_[off / 4]; }

  void wait_done() override {
    if (!irq_) {
      while (!(read32(regmap::CTRL) & regmap::AP_DONE)) {
      }
      return;
    }
    uint32_t count = 0;
    if (::read(uio_fd_, &count, sizeof count) != ssize_t(sizeof count))
      throw sys_error("attente de l'interruption UIO");
    write32(regmap::ISR, 1);  // acquitte ap_done
    (void)read32(regmap::CTRL);  // efface ap_done (COR)
    unmask();
  }

  // Démasque l'interruption : avant le premier ap_start, puis après chaque fin.
  void unmask() {
    if (!irq_) return;
    const uint32_t one = 1;
    if (::write(uio_fd_, &one, sizeof one) != ssize_t(sizeof one))
      throw sys_error("démasquage de l'interruption UIO");
  }

 private:
  bool irq_;
  std::string udmabuf_;
  int uio_fd_ = -1, dma_fd_ = -1;
  volatile uint32_t* regs_ = nullptr;
  Buffer buf_;
};

}  // namespace

std::unique_ptr<Device> make_uio_device(const DeviceOptions& o) {
  auto d = std::make_unique<UioDevice>(o);
  d->unmask();
  return d;
}

}  // namespace driver
