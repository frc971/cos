#include <filesystem>

#include "camera/camera_constants.h"
#include "camera/uvc_camera_node.h"
#include "gtest/gtest.h"
#include "unit_tests/test_helpers.h"
#include "utils/camera_utils.h"
#include "utils/constants_from_json.h"

namespace {

constexpr const char* kCameraConstantsPath =
    "/cos/constants/camera_constants.json";
constexpr const char* kConstantsRootPath = "/cos/constants";

TEST(CameraConstantsTest, ParsesValidCamerasAndSkipsInvalidEntries) {
  const auto path = cos_test::testing::WriteJsonFile(
      std::filesystem::temp_directory_path() / "camera_constants_test.json",
      {{"cameras",
        {{{"name", "front"},
          {"pipeline", "pipe"},
          {"intrinsics_path", "intrinsics.json"},
          {"extrinsics_path", "extrinsics.json"},
          {"backlight", 1.5},
          {"frame_width", 640},
          {"frame_height", 480},
          {"fps", 30},
          {"exposure", 10.0},
          {"brightness", 2.0},
          {"sharpness", 3.0},
          {"max_frame_size", 1234},
          {"max_payload_size", 5678},
          {"serial_id", "abc"},
          {"stream_ratio", 0.5},
          {"port", 1181},
          {"streamer_fps", 15},
          {"detector_type", "opencv_cpu"},
          {"camera_type", "uvc"}},
         nullptr,
         {{"name", "front"}, {"camera_type", "mipi"}},
         {{"name", "rear"},
          {"detector_type", "austin_gpu"},
          {"camera_type", "opencv"}}}}});

  const camera::camera_constants_t constants = camera::GetCameraConstants(path);

  ASSERT_EQ(constants.size(), 2);
  const auto& front = constants.at("front");
  EXPECT_EQ(front.name, "front");
  EXPECT_EQ(front.pipeline, "pipe");
  EXPECT_EQ(front.intrinsics_path, "intrinsics.json");
  EXPECT_EQ(front.extrinsics_path, "extrinsics.json");
  EXPECT_EQ(front.backlight, 1.5);
  EXPECT_EQ(front.frame_width, 640U);
  EXPECT_EQ(front.frame_height, 480U);
  EXPECT_EQ(front.fps, 30U);
  EXPECT_EQ(front.exposure, 10.0);
  EXPECT_EQ(front.brightness, 2.0);
  EXPECT_EQ(front.sharpness, 3.0);
  EXPECT_EQ(front.max_frame_size, 1234U);
  EXPECT_EQ(front.max_payload_size, 5678U);
  EXPECT_EQ(front.serial_id, "abc");
  EXPECT_EQ(front.stream_ratio, 0.5);
  EXPECT_EQ(front.port, 1181U);
  EXPECT_EQ(front.streamer_fps, 15U);
  EXPECT_EQ(front.detector_type, camera::DetectorType::OPENCV_CPU);
  EXPECT_EQ(front.camera_type, camera::CameraType::UVC);

  EXPECT_EQ(constants.at("rear").detector_type,
            camera::DetectorType::AUSTIN_GPU);
  EXPECT_EQ(constants.at("rear").camera_type, camera::CameraType::OPENCV);
}

TEST(CameraConstantTest, SortsByName) {
  camera::camera_constant_t a{.name = "a"};
  camera::camera_constant_t b{.name = "b"};

  EXPECT_TRUE(a < b);
  EXPECT_FALSE(b < a);
}

void ExpectIntrinsicsLoad(const std::string& path) {
  const nlohmann::json intrinsics = utils::ReadIntrinsics(path);

  for (const char* field :
       {"fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2"}) {
    SCOPED_TRACE(field);
    ASSERT_TRUE(intrinsics.contains(field));
    EXPECT_TRUE(intrinsics.at(field).is_number());
  }

  EXPECT_NO_THROW(
      (void)utils::CameraMatrixFromJson<Eigen::Matrix3d>(intrinsics));
  EXPECT_NO_THROW(
      (void)utils::DistortionCoefficientsFromJson<frc::apriltag::DistCoeffs>(
          intrinsics));
}

void ExpectExtrinsicsLoad(const std::string& path) {
  const nlohmann::json extrinsics = utils::ReadExtrinsics(path);

  for (const char* field : {"translation_x", "translation_y", "translation_z",
                            "rotation_x", "rotation_y", "rotation_z"}) {
    SCOPED_TRACE(field);
    ASSERT_TRUE(extrinsics.contains(field));
    EXPECT_TRUE(extrinsics.at(field).is_number());
  }

  EXPECT_NO_THROW((void)utils::ExtrinsicsJsonToCameraToRobot(extrinsics));
}

TEST(CameraConstantsTest, CheckedInUvcConstantsLoadCompletely) {
  const camera::camera_constants_t constants =
      camera::GetCameraConstants(kCameraConstantsPath);

  ASSERT_FALSE(constants.empty());
  for (const auto& [name, constant] : constants) {
    SCOPED_TRACE(name);
    EXPECT_EQ(constant.camera_type, camera::CameraType::UVC);
    EXPECT_EQ(constant.detector_type, camera::DetectorType::AUSTIN_GPU);
    ASSERT_TRUE(constant.intrinsics_path.has_value());
    ASSERT_TRUE(constant.extrinsics_path.has_value());
    ASSERT_TRUE(constant.frame_width.has_value());
    ASSERT_TRUE(constant.frame_height.has_value());
    ASSERT_TRUE(constant.fps.has_value());
    ASSERT_TRUE(constant.serial_id.has_value());
    ASSERT_TRUE(constant.max_frame_size.has_value());
    ASSERT_TRUE(constant.max_payload_size.has_value());
    ASSERT_TRUE(constant.stream_ratio.has_value());
    ASSERT_TRUE(constant.port.has_value());

    EXPECT_NO_THROW((void)camera::UVCCameraConfig(constant));
    ExpectIntrinsicsLoad(*constant.intrinsics_path);
    ExpectExtrinsicsLoad(*constant.extrinsics_path);
  }
}

TEST(CameraConstantsTest, EveryConstantsSubdirectoryHasLoadableCameras) {
  const camera::camera_constants_t constants =
      camera::GetCameraConstants(kCameraConstantsPath);

  size_t folders_checked = 0;
  size_t cameras_checked = 0;
  for (const std::filesystem::directory_entry& folder :
       std::filesystem::directory_iterator(kConstantsRootPath)) {
    if (!folder.is_directory()) {
      continue;
    }

    folders_checked++;
    const std::string folder_name = folder.path().filename().string();
    SCOPED_TRACE(folder_name);

    for (const std::filesystem::directory_entry& file :
         std::filesystem::directory_iterator(folder.path())) {
      if (!file.is_regular_file()) {
        continue;
      }

      const std::string stem = file.path().stem().string();
      constexpr std::string_view kIntrinsicsSuffix = "_intrinsics";
      if (!stem.ends_with(kIntrinsicsSuffix)) {
        continue;
      }

      const std::string camera_name =
          stem.substr(0, stem.size() - kIntrinsicsSuffix.size());
      const std::filesystem::path intrinsics_path = file.path();
      const std::filesystem::path extrinsics_path =
          folder.path() / (camera_name + "_extrinsics.json");
      const std::string constant_name = folder_name + "_" + camera_name;

      SCOPED_TRACE(constant_name);
      ASSERT_TRUE(std::filesystem::exists(extrinsics_path));
      ASSERT_TRUE(constants.contains(constant_name));

      const camera::camera_constant_t& constant = constants.at(constant_name);
      ASSERT_TRUE(constant.intrinsics_path.has_value());
      ASSERT_TRUE(constant.extrinsics_path.has_value());
      EXPECT_EQ(std::filesystem::path(*constant.intrinsics_path),
                intrinsics_path);
      EXPECT_EQ(std::filesystem::path(*constant.extrinsics_path),
                extrinsics_path);

      EXPECT_NO_THROW((void)camera::UVCCameraConfig(constant));
      ExpectIntrinsicsLoad(intrinsics_path.string());
      ExpectExtrinsicsLoad(extrinsics_path.string());
      cameras_checked++;
    }
  }

  EXPECT_GT(folders_checked, 0U);
  EXPECT_GT(cameras_checked, 0U);
}

}  // namespace
