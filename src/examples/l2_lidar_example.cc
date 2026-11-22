/**********************************************************************
 Copyright (c) 2020-2024, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/

#include <iostream>

#include "unitree_lidar_sdk.h"

int main() {
    constexpr char kLidarIp[] = "192.168.0.250";
    constexpr char kLocalIp[] = "192.168.0.36";
    constexpr unsigned short kLidarPort = 6101;
    constexpr unsigned short kLocalPort = 6201;

    using namespace unilidar_sdk2;
    UnitreeLidarReader* reader = createUnitreeLidarReader();
    if (reader->initializeUDP(kLidarPort, kLidarIp, kLocalPort, kLocalIp)) {
        std::cerr << "Failed to initialize LiDAR UDP on " << kLocalIp << ':'
                  << kLocalPort << '\n';
        return 1;
    }

    LidarImuData imu;
    PointCloudUnitree cloud;
    while (true) {
        switch (reader->runParse()) {
            case LIDAR_IMU_DATA_PACKET_TYPE:
                if (reader->getImuData(imu)) {
                    std::cout << "IMU seq=" << imu.info.seq
                              << " acceleration=(" << imu.linear_acceleration[0]
                              << ", " << imu.linear_acceleration[1]
                              << ", " << imu.linear_acceleration[2] << ')' << std::endl;
                }
                break;
            case LIDAR_POINT_DATA_PACKET_TYPE:
                if (reader->getPointCloud(cloud) && !cloud.points.empty()) {
                    const auto& point = cloud.points.front();
                    std::cout << "Cloud id=" << cloud.id
                              << " points=" << cloud.points.size()
                              << " first xyz=(" << point.x << ", " << point.y
                              << ", " << point.z << ')' << std::endl;
                }
                break;
            default:
                break;
        }
    }
}
