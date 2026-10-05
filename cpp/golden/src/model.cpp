#include "golden/model.hpp"

#include <fstream>
#include <stdexcept>

#include "golden/golden.hpp"
#include "golden/json.hpp"

namespace golden {

const char* layer_type_name(LayerType t) {
  switch (t) {
    case LayerType::Conv: return "conv";
    case LayerType::Maxpool: return "maxpool";
    case LayerType::Upsample: return "upsample";
    case LayerType::Route: return "route";
    case LayerType::Yolo: return "yolo";
    case LayerType::Region: return "region";
  }
  return "?";
}

static LayerType parse_type(const std::string& s) {
  if (s == "conv") return LayerType::Conv;
  if (s == "maxpool") return LayerType::Maxpool;
  if (s == "upsample") return LayerType::Upsample;
  if (s == "route") return LayerType::Route;
  if (s == "yolo") return LayerType::Yolo;
  if (s == "region") return LayerType::Region;
  throw std::runtime_error("manifest : type de couche inconnu " + s);
}

static BufRef parse_buf(const Json& j) {
  BufRef r;
  if (j.is_null()) return r;
  r.buf = j["buf"].str();
  r.offset = j["offset"].integer();
  return r;
}

static std::vector<int> ints(const Json& j) {
  std::vector<int> v;
  for (const auto& x : j.arr()) v.push_back(static_cast<int>(x.integer()));
  return v;
}

template <class T>
static std::vector<T> read_blob(const std::string& path, int64_t size) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("impossible d'ouvrir " + path);
  int64_t n = f.tellg();
  if (n != size)
    throw std::runtime_error(path + " : " + std::to_string(n) + " octets, " +
                             std::to_string(size) + " attendus");
  if (size % static_cast<int64_t>(sizeof(T)) != 0)
    throw std::runtime_error(path + " : taille non multiple de l'élément");
  // Little-endian, comme les hôtes visés (x86, ARM).
  std::vector<T> v(static_cast<size_t>(size) / sizeof(T));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), size);
  return v;
}

static void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("manifest : " + what);
}

std::vector<int> Model::heads() const {
  std::vector<int> h;
  for (const auto& l : layers)
    if (l.is_head()) h.push_back(l.id);
  return h;
}

Model Model::load(const std::string& dir) {
  const Json j = Json::parse_file(dir + "/manifest.json");
  Model m;
  m.format_version = static_cast<int>(j["format_version"].integer());
  require(m.format_version == manifest_format_version(), "version de format inattendue");
  m.network = j["network"].str();
  m.classes = static_cast<int>(j["classes"].integer());
  const auto& in = j["input"];
  auto shape = ints(in["shape"]);
  require(shape.size() == 4 && shape[0] == 1, "entrée (1, C, H, W) attendue");
  m.in_c = shape[1];
  m.in_h = shape[2];
  m.in_w = shape[3];
  m.input_scale = in["scale"].num();
  m.input = parse_buf(in["buf"]);
  for (const auto& a : j["anchors"].arr()) m.anchors.emplace_back(a[0].num(), a[1].num());
  for (const auto& kv : j["buffers"].obj()) m.buffers[kv.first] = kv.second.integer();

  for (const auto& e : j["layers"].arr()) {
    Layer l;
    l.id = static_cast<int>(e["id"].integer());
    require(l.id == static_cast<int>(m.layers.size()), "couches non numérotées dans l'ordre");
    l.type = parse_type(e["type"].str());
    auto os = ints(e["out_shape"]);
    require(os.size() == 3, "out_shape (C, H, W) attendu");
    l.out_c = os[0];
    l.out_h = os[1];
    l.out_w = os[2];
    if (e.has("in")) l.in = parse_buf(e["in"]);
    if (e.has("out")) l.out = parse_buf(e["out"]);
    switch (l.type) {
      case LayerType::Conv: {
        l.k = static_cast<int>(e["k"].integer());
        l.s = static_cast<int>(e["s"].integer());
        l.pad = static_cast<int>(e["pad"].integer());
        l.cin = static_cast<int>(e["cin"].integer());
        l.cout = static_cast<int>(e["cout"].integer());
        const std::string act = e["act"].str();
        require(act == "leaky" || act == "linear", "activation inconnue " + act);
        l.leaky = act == "leaky";
        if (e.has("fused_pool") && !e["fused_pool"].is_null()) {
          const auto& fp = e["fused_pool"];
          l.pool_layer = static_cast<int>(fp["layer"].integer());
          l.pool_k = static_cast<int>(fp["k"].integer());
          l.pool_s = static_cast<int>(fp["s"].integer());
        }
        if (e.has("prepool_out")) l.prepool_out = parse_buf(e["prepool_out"]);
        l.w_offset = e["w_offset"].integer();
        l.b_offset = e["b_offset"].integer();
        l.m0_offset = e["m0_offset"].integer();
        l.shift = static_cast<int>(e["shift"].integer());
        l.in_scale = e["in_scale"].num();
        l.out_scale = e["out_scale"].num();
        require(l.w_offset % BLOB_ALIGN == 0 && l.b_offset % BLOB_ALIGN == 0 &&
                    l.m0_offset % BLOB_ALIGN == 0,
                "offsets de blobs non alignés sur 64 octets");
        require(l.shift >= 1 && l.shift <= 31, "shift hors de [1, 31]");
        if (e.has("qmax")) l.qmax = static_cast<int>(e["qmax"].integer());
        require(l.qmax >= 1 && l.qmax <= 127, "qmax hors de [1, 127]");
        if (e.has("wbits")) l.wbits = static_cast<int>(e["wbits"].integer());
        require(l.wbits == 4 || l.wbits == 8, "wbits : 4 ou 8");
        break;
      }
      case LayerType::Maxpool:
        l.k = static_cast<int>(e["k"].integer());
        l.s = static_cast<int>(e["s"].integer());
        if (e.has("fused_into")) l.fused_into = static_cast<int>(e["fused_into"].integer());
        break;
      case LayerType::Upsample:
        l.s = static_cast<int>(e["s"].integer());
        l.scale = e["scale"].num();
        break;
      case LayerType::Route:
        l.from = ints(e["from"]);
        l.layout = e["layout"].str();
        l.scale = e["scale"].num();
        break;
      case LayerType::Yolo:
      case LayerType::Region:
        if (l.type == LayerType::Yolo) l.mask = ints(e["mask"]);
        else l.num = static_cast<int>(e["num"].integer());
        l.scale = e["scale"].num();
        l.lut_offset = e["lut_offset"].integer();
        l.exp_frac = static_cast<int>(e["exp_frac"].integer());
        break;
    }
    m.layers.push_back(l);
  }

  const auto& blobs = j["blobs"];
  m.weights = read_blob<int8_t>(dir + "/weights.bin", blobs["weights.bin"].integer());
  m.bias = read_blob<int32_t>(dir + "/bias.bin", blobs["bias.bin"].integer());
  m.m0 = read_blob<int32_t>(dir + "/requant.bin", blobs["requant.bin"].integer());
  m.luts = read_blob<uint32_t>(dir + "/luts.bin", blobs["luts.bin"].integer());

  // Poids 4 bits (T10.10) : quartet bas = poids d'indice pair, extension de signe ; copie
  // int8 ajoutée en fin de `weights` (alignée sur 64), sur laquelle pointe w_offset.
  const int64_t packed_size = int64_t(m.weights.size());
  for (auto& l : m.layers) {
    if (l.type != LayerType::Conv || l.wbits == 8) continue;
    const int64_t n = int64_t(l.cout) * l.cin * l.k * l.k;
    require(l.w_offset + (n + 1) / 2 <= packed_size, "poids hors de weights.bin");
    const int64_t dst = (int64_t(m.weights.size()) + BLOB_ALIGN - 1) / BLOB_ALIGN * BLOB_ALIGN;
    m.weights.resize(size_t(dst + n), 0);
    for (int64_t i = 0; i < n; ++i) {
      const uint8_t byte = uint8_t(m.weights[size_t(l.w_offset + i / 2)]);
      const int nib = (i % 2 == 0) ? (byte & 0xF) : (byte >> 4);
      m.weights[size_t(dst + i)] = int8_t(nib >= 8 ? nib - 16 : nib);
    }
    l.w_offset = dst;
  }

  // Contrôles de cohérence : blobs couverts, tampons assez grands.
  auto fits = [&](const BufRef& r, int64_t bytes) {
    require(m.buffers.count(r.buf) != 0, "tampon inconnu " + r.buf);
    require(r.offset >= 0 && r.offset + bytes <= m.buffers.at(r.buf),
            "tenseur hors du tampon " + r.buf);
  };
  fits(m.input, int64_t(m.in_c) * m.in_h * m.in_w);
  for (const auto& l : m.layers) {
    if (l.type != LayerType::Conv) continue;
    require(l.w_offset + int64_t(l.cout) * l.cin * l.k * l.k <= int64_t(m.weights.size()),
            "poids hors de weights.bin");
    require(l.b_offset / 4 + l.cout <= int64_t(m.bias.size()), "biais hors de bias.bin");
    require(l.m0_offset / 4 + l.cout <= int64_t(m.m0.size()), "M0 hors de requant.bin");
    fits(l.in, 0);
    fits(l.out, int64_t(l.out_c) * l.out_h * l.out_w);
    if (l.prepool_out.valid())
      fits(l.prepool_out, int64_t(l.cout) * l.out_h * l.pool_s * l.out_w * l.pool_s);
  }
  for (const auto& l : m.layers)
    if (l.is_head())
      require(l.lut_offset % 4 == 0 && l.lut_offset / 4 + 3 * 256 <= int64_t(m.luts.size()),
              "tables hors de luts.bin");
  return m;
}

}  // namespace golden
