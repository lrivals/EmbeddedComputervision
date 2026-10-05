// `hls::stream` : en-tête Vitis en synthèse, file FIFO minimale pour la C-sim avec g++
// (T9.4.2). En C-sim, DATAFLOW exécute les étages l'un après l'autre : une file non bornée
// suffit. `max_size` (T10.8) donne son occupation maximale ; elle vaut ici toute la sortie de
// l'étage producteur, pas la profondeur utile en matériel (stream_model.fifo_depths).
#pragma once

#if defined(__SYNTHESIS__) || defined(__VITIS_HLS__)
#include <hls_stream.h>
#else
#include <deque>
#include <stdexcept>

namespace hls {

template <class T>
class stream {
 public:
  stream() = default;
  explicit stream(const char*) {}
  void write(const T& v) {
    q_.push_back(v);
    if (q_.size() > max_) max_ = q_.size();
  }
  T read() {
    if (q_.empty()) throw std::runtime_error("hls::stream : lecture d'une file vide");
    T v = q_.front();
    q_.pop_front();
    return v;
  }
  bool empty() const { return q_.empty(); }
  size_t size() const { return q_.size(); }
  size_t max_size() const { return max_; }

 private:
  std::deque<T> q_;
  size_t max_ = 0;
};

}  // namespace hls
#endif
