#include "golden/json.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace golden {

class JsonParser {
 public:
  explicit JsonParser(const std::string& t) : t_(t) {}

  Json document() {
    Json v = value();
    ws();
    if (p_ != t_.size()) fail("caractères après la valeur");
    return v;
  }

 private:
  const std::string& t_;
  size_t p_ = 0;

  [[noreturn]] void fail(const std::string& what) const {
    throw std::runtime_error("JSON : " + what + " (position " + std::to_string(p_) + ")");
  }
  void ws() {
    while (p_ < t_.size() && (t_[p_] == ' ' || t_[p_] == '\n' || t_[p_] == '\r' || t_[p_] == '\t'))
      ++p_;
  }
  char peek() {
    ws();
    if (p_ >= t_.size()) fail("fin inattendue");
    return t_[p_];
  }
  void expect(char c) {
    if (peek() != c) fail(std::string("attendu '") + c + "'");
    ++p_;
  }
  bool literal(const char* w) {
    size_t n = std::char_traits<char>::length(w);
    if (t_.compare(p_, n, w) != 0) return false;
    p_ += n;
    return true;
  }

  Json value() {
    Json v;
    char c = peek();
    if (c == '{') {
      v.kind_ = Json::Kind::Object;
      ++p_;
      if (peek() == '}') { ++p_; return v; }
      for (;;) {
        std::string k = string();
        expect(':');
        v.o_[k] = value();
        if (peek() == ',') { ++p_; continue; }
        expect('}');
        return v;
      }
    }
    if (c == '[') {
      v.kind_ = Json::Kind::Array;
      ++p_;
      if (peek() == ']') { ++p_; return v; }
      for (;;) {
        v.a_.push_back(value());
        if (peek() == ',') { ++p_; continue; }
        expect(']');
        return v;
      }
    }
    if (c == '"') {
      v.kind_ = Json::Kind::String;
      v.s_ = string();
      return v;
    }
    if (literal("true")) { v.kind_ = Json::Kind::Bool; v.b_ = true; return v; }
    if (literal("false")) { v.kind_ = Json::Kind::Bool; return v; }
    if (literal("null")) return v;
    if (literal("NaN")) { v.kind_ = Json::Kind::Number; v.n_ = NAN; return v; }
    const char* begin = t_.c_str() + p_;
    char* end = nullptr;
    v.n_ = std::strtod(begin, &end);
    if (end == begin) fail("valeur invalide");
    p_ += static_cast<size_t>(end - begin);
    v.kind_ = Json::Kind::Number;
    return v;
  }

  std::string string() {
    expect('"');
    std::string out;
    while (p_ < t_.size() && t_[p_] != '"') {
      char c = t_[p_++];
      if (c != '\\') { out += c; continue; }
      if (p_ >= t_.size()) fail("échappement tronqué");
      char e = t_[p_++];
      switch (e) {
        case 'n': out += '\n'; break;
        case 't': out += '\t'; break;
        case 'r': out += '\r'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'u': {
          if (p_ + 4 > t_.size()) fail("\\u tronqué");
          unsigned cp = std::stoul(t_.substr(p_, 4), nullptr, 16);
          p_ += 4;
          // UTF-8 (sans paires de substitution : inutiles pour nos fichiers).
          if (cp < 0x80) {
            out += static_cast<char>(cp);
          } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
          } else {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
          }
          break;
        }
        default: out += e;
      }
    }
    if (p_ >= t_.size()) fail("chaîne non terminée");
    ++p_;
    return out;
  }
};

Json Json::parse(const std::string& text) { return JsonParser(text).document(); }

Json Json::parse_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("impossible d'ouvrir " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return parse(ss.str());
}

static void check(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(std::string("JSON : ") + what + " attendu");
}

bool Json::has(const std::string& key) const {
  return kind_ == Kind::Object && o_.count(key) != 0;
}
double Json::num() const { check(kind_ == Kind::Number, "nombre"); return n_; }
long long Json::integer() const {
  check(kind_ == Kind::Number && std::floor(n_) == n_, "entier");
  return static_cast<long long>(n_);
}
bool Json::boolean() const { check(kind_ == Kind::Bool, "booléen"); return b_; }
const std::string& Json::str() const { check(kind_ == Kind::String, "chaîne"); return s_; }
const std::vector<Json>& Json::arr() const { check(kind_ == Kind::Array, "tableau"); return a_; }
const std::map<std::string, Json>& Json::obj() const {
  check(kind_ == Kind::Object, "objet");
  return o_;
}
const Json& Json::operator[](const std::string& key) const {
  auto it = obj().find(key);
  if (it == o_.end()) throw std::runtime_error("JSON : clé absente « " + key + " »");
  return it->second;
}
size_t Json::size() const { return kind_ == Kind::Object ? o_.size() : arr().size(); }

}  // namespace golden
