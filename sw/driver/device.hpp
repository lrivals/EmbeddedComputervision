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
  // Cohérence : tampons non cachés en `uio` (O_SYNC), rien à faire en `sim`.
  virtual void sync_for_device(const Buffer&) {}
  virtual void sync_for_cpu(const Buffer&) {}

  // Second noyau `yolo_post` (T9.1), dans sa propre fenêtre de registres ; absent par défaut.
  virtual bool has_post() const { return false; }
  virtual void post_write32(uint32_t, uint32_t) { no_post(); }
  virtual uint32_t post_read32(uint32_t) { no_post(); }
  virtual void post_wait_done() { no_post(); }

 private:
  [[noreturn]] static void no_post();
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
  std::string uio_post;                // registres de yolo_post (T9.1) ; vide : absent en uio
};

std::unique_ptr<Device> make_device(const DeviceOptions& o);

#ifdef SW_HAVE_SIM
std::unique_ptr<Device> make_sim_device();  // yolo_conv et yolo_post
#endif
std::unique_ptr<Device> make_uio_device(const DeviceOptions& o);

}  // namespace driver
