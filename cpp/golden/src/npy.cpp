#include "golden/npy.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace golden {

size_t NpyInt8::numel() const {
  size_t n = 1;
  for (auto d : shape) n *= static_cast<size_t>(d);
  return n;
}

static std::string field(const std::string& header, const std::string& key) {
  auto k = header.find("'" + key + "'");
  if (k == std::string::npos) throw std::runtime_error("npy : champ absent " + key);
  auto colon = header.find(':', k);
  return header.substr(colon + 1);
}

NpyInt8 npy_load_int8(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("impossible d'ouvrir " + path);
  char magic[8];
  f.read(magic, 8);
  if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0)
    throw std::runtime_error("npy : signature invalide " + path);
  uint32_t hlen = 0;
  if (magic[6] == 1) {
    unsigned char b[2];
    f.read(reinterpret_cast<char*>(b), 2);
    hlen = b[0] | (b[1] << 8);
  } else {
    unsigned char b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    hlen = b[0] | (b[1] << 8) | (b[2] << 16) | (static_cast<uint32_t>(b[3]) << 24);
  }
  std::string header(hlen, '\0');
  f.read(&header[0], hlen);

  std::string descr = field(header, "descr");
  if (descr.find("i1") == std::string::npos)
    throw std::runtime_error("npy : int8 attendu dans " + path);
  if (field(header, "fortran_order").find("False") == std::string::npos)
    throw std::runtime_error("npy : ordre C attendu dans " + path);

  NpyInt8 out;
  std::string sh = field(header, "shape");
  sh = sh.substr(sh.find('(') + 1, sh.find(')') - sh.find('(') - 1);
  size_t p = 0;
  while (p < sh.size()) {
    while (p < sh.size() && (sh[p] == ' ' || sh[p] == ',')) ++p;
    if (p >= sh.size()) break;
    size_t q = p;
    while (q < sh.size() && sh[q] >= '0' && sh[q] <= '9') ++q;
    out.shape.push_back(std::stoll(sh.substr(p, q - p)));
    p = q;
  }
  out.data.resize(out.numel());
  f.read(reinterpret_cast<char*>(out.data.data()), static_cast<std::streamsize>(out.data.size()));
  if (!f) throw std::runtime_error("npy : données tronquées " + path);
  return out;
}

void npy_save_int8(const std::string& path, const std::vector<int64_t>& shape,
                   const int8_t* data) {
  std::string sh = "(";
  size_t n = 1;
  for (auto d : shape) {
    sh += std::to_string(d) + ", ";
    n *= static_cast<size_t>(d);
  }
  if (shape.size() > 1) sh.erase(sh.size() - 2);  // garder « (n,) » en dimension 1
  else if (!shape.empty()) sh.erase(sh.size() - 1);
  sh += ")";
  std::string header = "{'descr': '|i1', 'fortran_order': False, 'shape': " + sh + ", }";
  // En-tête v1 : 10 octets + texte, total multiple de 64, terminé par '\n'.
  size_t total = 10 + header.size() + 1;
  header.append((64 - total % 64) % 64, ' ');
  header += '\n';
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("impossible d'écrire " + path);
  f.write("\x93NUMPY\x01\x00", 8);
  unsigned char hl[2] = {static_cast<unsigned char>(header.size() & 0xFF),
                         static_cast<unsigned char>(header.size() >> 8)};
  f.write(reinterpret_cast<const char*>(hl), 2);
  f.write(header.data(), static_cast<std::streamsize>(header.size()));
  f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
}

}  // namespace golden
