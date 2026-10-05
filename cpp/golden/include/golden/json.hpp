// Lecteur JSON minimal pour manifest.json et detections.json (T5.1).
// Nombres lus par strtod : aller-retour exact des `repr` Python (17 chiffres significatifs).
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace golden {

class Json {
 public:
  enum class Kind { Null, Bool, Number, String, Array, Object };

  static Json parse(const std::string& text);
  static Json parse_file(const std::string& path);

  Kind kind() const { return kind_; }
  bool is_null() const { return kind_ == Kind::Null; }
  bool has(const std::string& key) const;

  // Accès typés : lèvent std::runtime_error si le type ne correspond pas.
  double num() const;
  long long integer() const;  // nombre entier exact
  bool boolean() const;
  const std::string& str() const;
  const std::vector<Json>& arr() const;
  const std::map<std::string, Json>& obj() const;
  const Json& operator[](const std::string& key) const;
  const Json& operator[](size_t i) const { return arr().at(i); }
  size_t size() const;

 private:
  friend class JsonParser;
  Kind kind_ = Kind::Null;
  bool b_ = false;
  double n_ = 0.0;
  std::string s_;
  std::vector<Json> a_;
  std::map<std::string, Json> o_;
};

}  // namespace golden
