#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include <wpi/struct/Struct.h>

namespace gamepiece {

// Field coordinates (meters), yaw (radians, modulo pi), 2-sigma semiaxes
// (meters), and the tracker's predicted visible ball density (balls/m^2).
using cluster_nt_t = struct ClusterNT {
  double center_x;
  double center_y;
  double field_relative_semimajor_axis_yaw;
  double semimajor_length;
  double semiminor_length;
  double predicted_density;
};

}  // namespace gamepiece

template <>
struct wpi::Struct<gamepiece::cluster_nt_t> {
  static constexpr auto GetTypeName() -> std::string_view { return "ClusterNT"; }
  static constexpr auto GetSize() -> std::size_t { return 48; }
  static constexpr auto GetSchema() -> std::string_view {
    return "double center_x;double center_y;"
           "double field_relative_semimajor_axis_yaw;double semimajor_length;"
           "double semiminor_length;double predicted_density";
  }

  static auto Unpack(std::span<const std::uint8_t> data)
      -> gamepiece::cluster_nt_t {
    return {.center_x = wpi::UnpackStruct<double, 0>(data),
            .center_y = wpi::UnpackStruct<double, 8>(data),
            .field_relative_semimajor_axis_yaw =
                wpi::UnpackStruct<double, 16>(data),
            .semimajor_length = wpi::UnpackStruct<double, 24>(data),
            .semiminor_length = wpi::UnpackStruct<double, 32>(data),
            .predicted_density = wpi::UnpackStruct<double, 40>(data)};
  }

  static void Pack(std::span<std::uint8_t> data,
                   const gamepiece::cluster_nt_t& value) {
    wpi::PackStruct<0>(data, value.center_x);
    wpi::PackStruct<8>(data, value.center_y);
    wpi::PackStruct<16>(data, value.field_relative_semimajor_axis_yaw);
    wpi::PackStruct<24>(data, value.semimajor_length);
    wpi::PackStruct<32>(data, value.semiminor_length);
    wpi::PackStruct<40>(data, value.predicted_density);
  }
};

static_assert(wpi::StructSerializable<gamepiece::cluster_nt_t>);
