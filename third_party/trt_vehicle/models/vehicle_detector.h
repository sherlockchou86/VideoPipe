#pragma once

#include <opencv2/core.hpp>
#include "detect_model.h"
#include "../util/algorithm_util.h"

namespace trt_vehicle {
    class VehicleDetector
    {
    private:
        /* data */
        DetectModel* vehicleModel = nullptr;
        std::vector<std::string> vehicle_labels = {"car", "bus", "truck"};
    public:
        VehicleDetector(const std::string& vehicleModelPath, float vehicleScoreThres = 0.5, float vehicleIouThres = 0.4);
        ~VehicleDetector();

        void detect(std::vector<cv::Mat>& img_datas, std::vector<std::vector<trt_vehicle::ObjBox>>& vehicles);
    };

}