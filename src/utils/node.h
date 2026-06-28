#pragma once
#include <functional>
#include <memory>
#include <vector>

#include "control_loops/context.h"

template <typename T>
class INode {
 public:
  virtual void RegisterCallback(
      const std::function<void(T, control_loops::MetaDataList,
                               std::shared_ptr<control_loops::Context>)>&
          callback) = 0;
  virtual ~INode() = default;
};
