#pragma once
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <typeindex>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace wpi::log { class DataLogWriter; }
namespace nt { class NetworkTableInstance; }
namespace control_loop { class IMessage; }
namespace logging {
using LogFunction = std::move_only_function<bool(const control_loop::IMessage&)>;
template <typename T>
auto RegisterFields(wpi::log::DataLogWriter&, const nt::NetworkTableInstance&,
                    std::string_view,
                    std::vector<std::string>&) -> LogFunction;
}

namespace control_loop {

class IMessage {
 public:
  virtual ~IMessage() = default;
  IMessage() = default;

  // Interface
  virtual auto GetType() -> const std::type_info& = 0;
  virtual auto GetSize() -> std::size_t = 0;
};

template <typename T>
class ValueMessage final : public IMessage {
 public:
  explicit ValueMessage(T value) : value(std::move(value)) {}
  auto GetType() -> const std::type_info& override {
    return typeid(ValueMessage<T>);
  }
  auto GetSize() -> std::size_t override { return sizeof(*this); }

  T value;
};

class MessageDescriptor {
 public:
  using RegistrationFunction = logging::LogFunction (*)(
      wpi::log::DataLogWriter&, const nt::NetworkTableInstance&, std::string_view,
      std::vector<std::string>&);
  MessageDescriptor(std::string_view channel, std::type_index type)
      : channel_(channel), types_({type}) {}
  MessageDescriptor(std::string_view channel,
                    std::unordered_set<std::type_index> types)
      : channel_(channel), types_(std::move(types)) {}

  template <typename T>
  static auto Publication(std::string_view channel)
      -> MessageDescriptor {
    using Stored = std::conditional_t<std::is_base_of_v<IMessage, T>, T,
                                      ValueMessage<T>>;
    MessageDescriptor descriptor(channel, typeid(Stored));
    if constexpr (requires { T::RegisterWPILog; }) {
      descriptor.registration_ = T::RegisterWPILog;
    } else if constexpr (std::is_arithmetic_v<T> ||
                         std::is_same_v<T, std::string>) {
      descriptor.registration_ = &logging::RegisterFields<T>;
    } else {
      descriptor.registration_ = nullptr;
    }
    return descriptor;
  }

  [[nodiscard]] auto GetChannel() const -> const std::string& {
    return channel_;
  }
  [[nodiscard]] auto GetTypes() const
      -> const std::unordered_set<std::type_index>& {
    return types_;
  }
  [[nodiscard]] auto GetRegistration() const
      -> const std::optional<RegistrationFunction>& {
    return registration_;
  }

 private:
  std::string channel_;
  std::unordered_set<std::type_index> types_;
  std::optional<RegistrationFunction> registration_;
};

}  // namespace control_loop
