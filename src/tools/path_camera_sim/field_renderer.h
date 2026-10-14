#pragma once

#include <filesystem>
#include <array>
#include <cstdint>
#include <vector>

#include <EGL/egl.h>
#include <Eigen/Core>
#include <GLES3/gl3.h>
#include <opencv2/core.hpp>

namespace path_camera_sim {

class FieldRenderer {
 public:
  FieldRenderer(const std::filesystem::path& field_directory,
                const cv::Size& image_size, const cv::Matx33d& intrinsics);
  ~FieldRenderer();

  FieldRenderer(const FieldRenderer&) = delete;
  auto operator=(const FieldRenderer&) -> FieldRenderer& = delete;

  auto Render(const Eigen::Matrix4d& world_to_camera) -> cv::Mat;
  void SetFuelLayout(double entropy, uint32_t seed);
  auto FuelPositions() const -> const std::vector<std::array<double, 3>>&;

 private:
  struct Draw {
    Eigen::Matrix4f model;
    cv::Vec4f color;
    GLint position_offset;
    GLint normal_offset;
    GLsizei vertex_stride;
    GLsizei index_count;
    GLenum index_type;
    size_t index_offset;
    bool transparent;
  };

  cv::Size image_size_;
  Eigen::Matrix4f projection_;
  std::vector<Draw> draws_;
  std::vector<Draw> staged_fuel_draws_;
  std::vector<Draw> active_fuel_draws_;
  std::vector<std::array<double, 3>> fuel_positions_;
  float field_length_ = 0;
  float field_width_ = 0;
  EGLDisplay display_ = EGL_NO_DISPLAY;
  EGLSurface surface_ = EGL_NO_SURFACE;
  EGLContext context_ = EGL_NO_CONTEXT;
  GLuint vertex_buffer_ = 0;
  GLuint frame_buffer_ = 0;
  GLuint color_buffer_ = 0;
  GLuint depth_buffer_ = 0;
  GLuint program_ = 0;
};

}  // namespace path_camera_sim
