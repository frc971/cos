#pragma once

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <NvBuffer.h>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <wpi/math/geometry/Pose3d.hpp>

#include "camera/nvjpeg_decode_node.h"
#include "gamepiece/gamepiece_detection.h"
#include "utils/node.h"
#include "yolo/yolo.h"

namespace gamepiece {

class YoloNode : public INode<std::vector<gamepiece_detection_t>> {
 public:
  YoloNode(const std::string& model_path,
           const std::vector<std::string>& class_names,
           const nlohmann::json& intrinsics, const nlohmann::json& extrinsics);
  ~YoloNode() override;

  void RegisterCallback(
      const std::function<
          void(std::vector<gamepiece_detection_t>, control_loops::MetaDataList,
               std::shared_ptr<control_loops::Context>)>& callback) override;

  void Detect(const std::shared_ptr<camera::DecodedJpegNvBuffer>& frame,
              control_loops::MetaDataList metadata,
              std::shared_ptr<control_loops::Context> ctx);

 private:
  void RunDetection(const std::shared_ptr<camera::DecodedJpegNvBuffer>& frame,
                    control_loops::MetaDataList metadata,
                    std::shared_ptr<control_loops::Context> ctx);

  static auto NvBufferToMat(NvBuffer* nv_buffer) -> cv::Mat;
  auto ComputePose(const cv::Rect& bbox) const -> wpi::math::Pose3d;

  yolo::Yolo yolo_;
  const std::vector<std::string> class_names_;

  float cam_cx_ = 0.0F;
  float cam_cy_ = 0.0F;
  float fx_ = 0.0F;
  float fy_ = 0.0F;
  float cam_pitch_ = 0.0F;
  float pinhole_height_ = 0.0F;
  wpi::math::Pose3d cam_pose_;

  std::vector<std::function<void(std::vector<gamepiece_detection_t>,
                                 control_loops::MetaDataList,
                                 std::shared_ptr<control_loops::Context>)>>
      callbacks_;

  std::mutex mutex_;
  std::condition_variable_any cv_;
  std::queue<std::function<void()>> tasks_;
  std::jthread worker_thread_;
};

}  // namespace gamepiece
