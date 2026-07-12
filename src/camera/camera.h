#pragma once

#include <cstdlib>
#include <functional>
#include <memory>

namespace camera {

class JpegBuffer {
 public:
  explicit JpegBuffer(size_t size) : size_(size), ptr_(std::malloc(size)) {}
  auto ptr() -> void* const { return ptr_; }

  auto size() -> size_t const { return size_; }
  ~JpegBuffer() { std::free(ptr_); }

 private:
  size_t size_;
  void* ptr_;
};

using CameraCallback =
    std::function<void(std::shared_ptr<JpegBuffer>, unsigned long timestamp)>;

class ICamera {
 public:
  virtual void RegisterCallback(const CameraCallback& callback) = 0;
  virtual void Start() = 0;
  virtual ~ICamera() = default;
};

}  // namespace camera
