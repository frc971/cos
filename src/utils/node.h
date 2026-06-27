#pragma once
#include <functional>

template <typename T>
class INode {
 public:
  virtual void RegisterCallback(const std::function<void(T)>& callback) = 0;
  virtual ~INode() = default;
};
