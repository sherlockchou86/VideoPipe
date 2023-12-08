
#pragma once

#include "class_model.h"

namespace trt_vehicle {
    class VehicleColorClassifier
    {
    private:
        std::vector<std::string> colors_ = {"black", "blue", "grey", "other", "red", "white", "yellow"};
        ClassModel* vehicleColorModel = nullptr;
    public:
        VehicleColorClassifier(const std::string& vehicleColorModelPath);
        ~VehicleColorClassifier();

        void classify(std::vector<cv::Mat>& img_datas, std::vector<trt_vehicle::ObjCls>& colors);
    };

}