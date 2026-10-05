// Backend `uio` (KV260, Ubuntu Kria) : registres par /dev/uioN, tampon contigu par u-dma-buf.
//
// - Le dtbo (hw/boards/kv260/pl.dtsi) déclare `s_axi_control` en `generic-uio` avec
//   l'interruption du noyau, et un nœud `u-dma-buf` (CMA) assez grand pour arène + poids +
//   paramètres.
// - Les ports HP ne sont pas cohérents avec les caches de l'ARM. Par défaut, le u-dma-buf est
//   ouvert en O_SYNC (non caché), ce qui rend `sync_for_*` inutiles au prix d'accès CPU plus
//   lents. Option `cached` (T10.6) : ouverture sans O_SYNC (caché), et `sync_for_*` vident ou
//   invalident la plage demandée par le sysfs de u-dma-buf (sync_offset, sync_size,
//   sync_direction, puis sync_for_device / sync_for_cpu).
// - Interruption UIO : écrire 1 dans le fd la démasque, `read` bloque jusqu'à la suivante ;
//   l'ISR du noyau HLS est acquittée par écriture de 1.
// - `yolo_post` (T9.1, option `uio_post`) : second nœud UIO, même protocole.
// - Streaming (T10.9, options `uio_stream` et `uio_dma`) : `yolo_stream` (même protocole) et
//   son AXI DMA, dont l'interruption S2MM (fin de la tête) est acquittée dans S2MM_DMASR.
//   Bitstream sans `yolo_conv` : `--uio none`. Non vérifié sur la carte.
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>

#include "device.hpp"
#include "post_regmap.hpp"
#include "regmap.hpp"
#include "stream_regmap.hpp"

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

// Fenêtre de registres d'un noyau HLS exposée par un nœud generic-uio.
struct UioRegs {
  int fd = -1;
  volatile uint32_t* regs = nullptr;
  bool irq = true;

  void open(const std::string& path, const char* kernel, bool use_irq) {
    irq = use_irq;
    fd = ::open(path.c_str(), O_RDWR | O_SYNC);
    if (fd < 0) throw sys_error("ouverture de " + path);
    void* p = ::mmap(nullptr, regmap::SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) throw sys_error("mmap des registres");
    regs = static_cast<volatile uint32_t*>(p);
    if (!(regs[regmap::CTRL / 4] & regmap::AP_IDLE))
      throw std::runtime_error(std::string(kernel) + " n'est pas au repos (bitstream chargé ?)");
    regs[regmap::GIE / 4] = irq ? 1 : 0;
    regs[regmap::IER / 4] = irq ? 1 : 0;
  }

  void close() {
    if (regs) ::munmap(const_cast<uint32_t*>(regs), regmap::SPAN);
    if (fd >= 0) ::close(fd);
    regs = nullptr;
    fd = -1;
  }

  void wait_done() {
    if (!irq) {
      while (!(regs[regmap::CTRL / 4] & regmap::AP_DONE)) {
      }
      return;
    }
    uint32_t count = 0;
    if (::read(fd, &count, sizeof count) != ssize_t(sizeof count))
      throw sys_error("attente de l'interruption UIO");
    regs[regmap::ISR / 4] = 1;    // acquitte ap_done
    (void)regs[regmap::CTRL / 4];  // efface ap_done (COR)
    unmask();
  }

  // Démasque l'interruption : avant le premier ap_start, puis après chaque fin.
  void unmask() {
    if (!irq) return;
    const uint32_t one = 1;
    if (::write(fd, &one, sizeof one) != ssize_t(sizeof one))
      throw sys_error("démasquage de l'interruption UIO");
  }
};

// Fenêtre de l'AXI DMA (PG021) : interruption de fin du S2MM.
struct UioDma {
  int fd = -1;
  volatile uint32_t* regs = nullptr;
  bool irq = true;

  void open(const std::string& path, bool use_irq) {
    using namespace dma_regmap;
    irq = use_irq;
    fd = ::open(path.c_str(), O_RDWR | O_SYNC);
    if (fd < 0) throw sys_error("ouverture de " + path);
    void* p = ::mmap(nullptr, SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) throw sys_error("mmap des registres du DMA");
    regs = static_cast<volatile uint32_t*>(p);
    regs[MM2S_DMACR / 4] = CR_RESET;  // remet les deux canaux à zéro
    while (regs[MM2S_DMACR / 4] & CR_RESET) {
    }
  }

  void close() {
    if (regs) ::munmap(const_cast<uint32_t*>(regs), dma_regmap::SPAN);
    if (fd >= 0) ::close(fd);
    regs = nullptr;
    fd = -1;
  }

  void wait_done() {
    using namespace dma_regmap;
    if (irq) {
      uint32_t count = 0;
      if (::read(fd, &count, sizeof count) != ssize_t(sizeof count))
        throw sys_error("attente de l'interruption du DMA");
    } else {
      while (!(regs[S2MM_DMASR / 4] & (SR_IOC_IRQ | SR_ERR))) {
      }
    }
    const uint32_t sr = regs[S2MM_DMASR / 4];
    regs[S2MM_DMASR / 4] = SR_IOC_IRQ | SR_ERR_IRQ;
    regs[MM2S_DMASR / 4] = SR_IOC_IRQ | SR_ERR_IRQ;
    unmask();
    if (sr & SR_ERR) throw std::runtime_error("DMA : erreur S2MM (DMASR)");
  }

  void unmask() {
    if (!irq || fd < 0) return;
    const uint32_t one = 1;
    if (::write(fd, &one, sizeof one) != ssize_t(sizeof one))
      throw sys_error("démasquage de l'interruption du DMA");
  }
};

class UioDevice : public Device {
 public:
  explicit UioDevice(const DeviceOptions& o)
      : irq_(o.irq), cached_(o.cached), udmabuf_(o.udmabuf) {
    if (o.uio != "none") conv_.open(o.uio, "yolo_conv", irq_);
    if (!o.uio_post.empty()) post_.open(o.uio_post, "yolo_post", irq_);
    if (o.uio_stream.empty() != o.uio_dma.empty())
      throw std::runtime_error("streaming : --uio-stream et --uio-dma vont ensemble");
    if (!o.uio_stream.empty()) {
      // Pas d'interruption sur yolo_stream : la fin de la tête est celle du S2MM.
      stream_.open(o.uio_stream, "yolo_stream", false);
      dma_.open(o.uio_dma, irq_);
    }
  }

  ~UioDevice() override {
    if (buf_.virt) ::munmap(buf_.virt, buf_.size);
    if (dma_fd_ >= 0) ::close(dma_fd_);
    dma_.close();
    stream_.close();
    post_.close();
    conv_.close();
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
    dma_fd_ = ::open(dev.c_str(), cached_ ? O_RDWR : O_RDWR | O_SYNC);
    if (dma_fd_ < 0) throw sys_error("ouverture de " + dev);
    void* p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, dma_fd_, 0);
    if (p == MAP_FAILED) throw sys_error("mmap de " + dev);
    buf_ = Buffer{static_cast<uint8_t*>(p), phys, size};
    return buf_;
  }

  void sync_for_device(const Buffer& b, size_t off, size_t len) override {
    sync(b, off, len, 1, "sync_for_device");
  }
  void sync_for_cpu(const Buffer& b, size_t off, size_t len) override {
    sync(b, off, len, 2, "sync_for_cpu");
  }

  void write32(uint32_t off, uint32_t v) override { conv().regs[off / 4] = v; }
  uint32_t read32(uint32_t off) override { return conv().regs[off / 4]; }
  void wait_done() override { conv().wait_done(); }

  bool has_post() const override { return post_.regs != nullptr; }
  void post_write32(uint32_t off, uint32_t v) override {
    if (!has_post()) Device::post_write32(off, v);
    post_.regs[off / 4] = v;
  }
  uint32_t post_read32(uint32_t off) override {
    if (!has_post()) return Device::post_read32(off);
    return post_.regs[off / 4];
  }
  void post_wait_done() override {
    if (!has_post()) Device::post_wait_done();
    post_.wait_done();
  }

  bool has_stream() const override { return stream_.regs != nullptr; }
  void stream_write32(uint32_t off, uint32_t v) override {
    if (!has_stream()) Device::stream_write32(off, v);
    stream_.regs[off / 4] = v;
  }
  uint32_t stream_read32(uint32_t off) override {
    if (!has_stream()) return Device::stream_read32(off);
    return stream_.regs[off / 4];
  }
  void dma_write32(uint32_t off, uint32_t v) override {
    if (!has_stream()) Device::dma_write32(off, v);
    dma_.regs[off / 4] = v;
  }
  uint32_t dma_read32(uint32_t off) override {
    if (!has_stream()) return Device::dma_read32(off);
    return dma_.regs[off / 4];
  }
  void dma_wait_done() override {
    if (!has_stream()) Device::dma_wait_done();
    dma_.wait_done();
  }

  void unmask() {
    if (conv_.regs) conv_.unmask();
    if (has_post()) post_.unmask();
    if (has_stream()) dma_.unmask();
  }

 private:
  UioRegs& conv() {
    if (!conv_.regs) throw std::runtime_error("yolo_conv absent (--uio none)");
    return conv_;
  }

  // direction : 1 vers le noyau (vidage), 2 vers l'ARM (invalidation).
  void sync(const Buffer& b, size_t off, size_t len, int direction, const char* what) {
    if (!cached_ || len == 0) return;
    if (b.virt != buf_.virt || off + len > buf_.size)
      throw std::runtime_error("uio : plage de synchronisation hors du u-dma-buf");
    const std::lock_guard<std::mutex> lock(sync_mutex_);  // séquence sysfs atomique
    const std::string sys = "/sys/class/u-dma-buf/" + udmabuf_ + "/";
    write_sysfs(sys + "sync_offset", std::to_string(off));
    write_sysfs(sys + "sync_size", std::to_string(len));
    write_sysfs(sys + "sync_direction", std::to_string(direction));
    write_sysfs(sys + what, "1");
  }

  static void write_sysfs(const std::string& path, const std::string& v) {
    std::ofstream f(path);
    if (!(f << v) || !f.flush()) throw std::runtime_error("écriture impossible : " + path);
  }

  bool irq_;
  bool cached_;
  std::mutex sync_mutex_;
  std::string udmabuf_;
  int dma_fd_ = -1;
  UioRegs conv_, post_, stream_;
  UioDma dma_;
  Buffer buf_;
};

}  // namespace

std::unique_ptr<Device> make_uio_device(const DeviceOptions& o) {
  auto d = std::make_unique<UioDevice>(o);
  d->unmask();
  return d;
}

}  // namespace driver
