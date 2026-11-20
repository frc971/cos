#pragma once

#include "control_loop/node.h"

class ExampleMessage final : public control_loop::IMessage {
 public:
  ExampleMessage(int example_data) : example_data(example_data) {}
  auto GetType() -> const std::type_info& override {
    return typeid(ExampleMessage);
  }
  auto GetSize() -> std::size_t override { return sizeof(int); }
  int example_data;
};

class ExampleNode final : public control_loop::INode {
 public:
  ExampleNode();
  ~ExampleNode() override;
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  void Callback(const control_loop::Context& context);
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override;

 private:
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
};
