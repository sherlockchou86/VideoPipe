
#include "vehicle_feature_encoder.h"


namespace trt_vehicle {
    
    VehicleFeatureEncoder::VehicleFeatureEncoder(const std::string& vehicleFeatureModelPath) {
        vehicleFeatureModel = new FeatureModel(vehicleFeatureModelPath);
    }
    
    VehicleFeatureEncoder::~VehicleFeatureEncoder()
    {
        if (vehicleFeatureModel != nullptr) {
            /* code */
            delete vehicleFeatureModel;
        }
        
    }
    
    void VehicleFeatureEncoder::encode(std::vector<cv::Mat>& img_datas, std::vector<std::vector<float>>& features) {
        features.clear();
        for (int i = 0; i < img_datas.size(); i++) {
            /* code */
            auto out = vehicleFeatureModel->predictNoPadding({img_datas[i]}).front();
            features.push_back(out);
        }
    }
}