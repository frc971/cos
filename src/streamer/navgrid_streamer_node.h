#pragma once

#include <filesystem>
#include <fstream>
#include <opencv4/opencv2/core/mat.hpp>
#include <stdexcept>
#include <vector>

#include <nadjieb/mjpeg_streamer.hpp>
#include <nlohmann/json.hpp>

#include "control_loop/node.h"

namespace streamer {

class NavgridMessage final : public control_loop::IMessage {
 public:
  explicit NavgridMessage(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open navgrid file: " + path.string());
    }
    const auto json = nlohmann::json::parse(file);
    field_size_x = json.at("field_size").at("x").get<double>();
    field_size_y = json.at("field_size").at("y").get<double>();
    node_size_meters = json.at("nodeSizeMeters").get<double>();
    navgrid = json.at("grid").get<std::vector<std::vector<bool>>>();
  }

  auto GetType() -> const std::type_info& override {
    return typeid(NavgridMessage);
  }
  auto GetSize() -> size_t override { return sizeof(*this); }

  double field_size_x;
  double field_size_y;
  double node_size_meters;
  std::vector<std::vector<bool>> navgrid;
};

class NavgridStreamerNode final : public control_loop::INode {
 public:
  NavgridStreamerNode(std::string_view input_path, std::string_view path,
                      int port,
                      const std::filesystem::path& field_image_path =
                          "/root/constants/assets/field.png");
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override;

 private:
  std::string input_path_;
  std::string path_;
  int port_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  cv::Mat field_image_;
  std::vector<unsigned char> jpeg_buffer_;
  nadjieb::MJPEGStreamer streamer_;
};

}  // namespace streamer
