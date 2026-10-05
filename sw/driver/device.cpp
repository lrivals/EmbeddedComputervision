#include "device.hpp"

#include <stdexcept>

namespace driver {

void Device::no_post() {
  throw std::runtime_error("yolo_post absent (--uio-post /dev/uioN en uio)");
}

std::unique_ptr<Device> make_device(const DeviceOptions& o) {
  if (o.backend == "uio") return make_uio_device(o);
#ifdef SW_HAVE_SIM
  if (o.backend == "sim") return make_sim_device();
#endif
  throw std::runtime_error("backend inconnu ou non compilé : " + o.backend +
                           " (SW_BACKEND=sim pour le backend sim)");
}

}  // namespace driver
