#pragma once

#include "class_model.h"

namespace trt_vehicle {
    class VehicleTypeClassifier
    {
    private:
        /* data */
        std::vector<std::string> types_ = {"bus", "business_car", "construction_truck", "large_truck", "sedan", "small_truck", "suv", "tanker", "van"};
        ClassModel* vehicleTypeModel = nullptr;
    public:
        VehicleTypeClassifier(const std::string& vehicleColorModelPath);
        ~VehicleTypeClassifier();

        void classify(std::vector<cv::Mat>& img_datas, std::vector<trt_vehicle::ObjCls>& types);
    };

}