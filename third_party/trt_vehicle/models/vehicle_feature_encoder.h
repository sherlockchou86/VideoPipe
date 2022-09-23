#pragma once
#include "feature_model.h"

namespace trt_vehicle {
    class VehicleFeatureEncoder
    {
    private:
        /* data */
        FeatureModel* vehicleFeatureModel = nullptr;
    public:
        VehicleFeatureEncoder(const std::string& vehicleFeatureModelPath);
        ~VehicleFeatureEncoder();

        void encode(std::vector<cv::Mat>& img_datas, std::vector<std::vector<float>>& features);
    };

}