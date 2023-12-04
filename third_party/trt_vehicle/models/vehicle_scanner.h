#pragma once


#include <opencv2/core.hpp>
#include "detect_model.h"
#include "../util/algorithm_util.h"

namespace trt_vehicle {
    class VehicleScanner
    {
    private:
        /* data */
        DetectModel* scannerModel = nullptr;
        std::vector<std::string> partLabels = {"truck",
                                                    "bus",
                                                    "car",
                                                    "suv",
                                                    "van",
                                                    "nowindow",
                                                    "iveco",
                                                    "pickup",
                                                    "wheel"};
    public:
        VehicleScanner(const std::string& scanModelPath, float scoreThres = 0.5, float iouThres = 0.4);
        ~VehicleScanner();

        void detect(std::vector<cv::Mat>& img_datas, std::vector<std::vector<trt_vehicle::ObjBox>>& parts);
    };

}
