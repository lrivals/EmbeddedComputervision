// Accès matériel minimal du driver (T7.2) : registres AXI-Lite du noyau, mémoire contiguë
// visible par ses ports m_axi, attente de fin.
//
// Deux réalisations :
// - `uio` (KV260, Ubuntu) : registres par /dev/uioN, tampon contigu par /dev/udmabufN ;
// - `sim` (PC) : registres émulés ; l'écriture d'ap_start retraduit les registres en
//   pointeurs et en `LayerDesc` puis appelle le noyau C-sim. Valide l'encodage des registres
//   et la traduction d'adresses physiques, pas seulement le noyau.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace driver {

// Zone contiguë : `virt` pour l'ARM, `phys` pour les ports m_axi du noyau.
struct Buffer {
  uint8_t* virt = nullptr;
  uint64_t phys = 0;
  size_t size = 0;
};

class Device {
 public:
  virtual ~Device() = default;
  virtual const char* name() const = 0;
  // Un seul tampon par appel ; en `uio`, c'est le u-dma-buf entier (taille ≤ celle du module).
  virtual Buffer alloc(size_t size) = 0;
  virtual void write32(uint32_t off, uint32_t v) = 0;
  virtual uint32_t read32(uint32_t off) = 0;
  // Après ap_start : rend la main quand ap_done est levé (interruption ou sondage).
  virtual void wait_done() = 0;
  // Cohérence d'une plage [off, off + len) du tampon : rien à faire en `sim` ni en `uio` non
  // caché (O_SYNC) ; en `uio --cached` (T10.6), vidage ou invalidation des caches de l'ARM.
  // Appelables depuis plusieurs threads (pipeline, T10.5).
  virtual void sync_for_device(const Buffer&, size_t /*off*/, size_t /*len*/) {}
  virtual void sync_for_cpu(const Buffer&, size_t /*off*/, size_t /*len*/) {}
  void sync_for_device(const Buffer& b) { sync_for_device(b, 0, b.size); }
  void sync_for_cpu(const Buffer& b) { sync_for_cpu(b, 0, b.size); }

  // Second noyau `yolo_post` (T9.1), dans sa propre fenêtre de registres ; absent par défaut.
  virtual bool has_post() const { return false; }
  virtual void post_write32(uint32_t, uint32_t) { no_post(); }
  virtual uint32_t post_read32(uint32_t) { no_post(); }
  virtual void post_wait_done() { no_post(); }

  // Streaming (T10.9) : noyau `yolo_stream` et AXI DMA, chacun dans sa fenêtre de registres
  // (stream_regmap.hpp) ; absents par défaut. `dma_wait_done` : fin du S2MM (tête reçue).
  virtual bool has_stream() const { return false; }
  virtual void stream_write32(uint32_t, uint32_t) { no_stream(); }
  virtual uint32_t stream_read32(uint32_t) { no_stream(); }
  virtual void dma_write32(uint32_t, uint32_t) { no_stream(); }
  virtual uint32_t dma_read32(uint32_t) { no_stream(); }
  virtual void dma_wait_done() { no_stream(); }

 private:
  [[noreturn]] static void no_post();
  [[noreturn]] static void no_stream();
};

struct DeviceOptions {
#ifdef SW_HAVE_SIM
  std::string backend = "sim";         // sim | uio
#else
  std::string backend = "uio";
#endif
  std::string uio = "/dev/uio0";       // registres de yolo_conv (nœud generic-uio du dtbo)
  std::string udmabuf = "udmabuf0";    // /dev/<nom>, /sys/class/u-dma-buf/<nom>/phys_addr
  bool irq = true;                     // uio : interruption ; false : sondage d'ap_done
  bool cached = false;                 // uio : u-dma-buf caché + sync explicites (T10.6)
  std::string uio_post;                // registres de yolo_post (T9.1) ; vide : absent en uio
  std::string uio_stream;              // registres de yolo_stream (T10.9) ; vide : absent
  std::string uio_dma;                 // registres de l'AXI DMA du streaming (T10.9)
};

std::unique_ptr<Device> make_device(const DeviceOptions& o);

#ifdef SW_HAVE_SIM
std::unique_ptr<Device> make_sim_device();  // yolo_conv, yolo_post, yolo_stream + AXI DMA
#endif
std::unique_ptr<Device> make_uio_device(const DeviceOptions& o);

}  // namespace driver
