
#include "vehicle_type_classifier.h"

namespace trt_vehicle {
        
    VehicleTypeClassifier::VehicleTypeClassifier(const std::string& vehicleTypeModelPath) {
        vehicleTypeModel = new ClassModel(vehicleTypeModelPath, false);
    }
    
    VehicleTypeClassifier::~VehicleTypeClassifier()
    {
        if (vehicleTypeModel != nullptr) {
            /* code */
            delete vehicleTypeModel;
        }
        
    }
    
    void VehicleTypeClassifier::classify(std::vector<cv::Mat>& img_datas, std::vector<trt_vehicle::ObjCls>& types) {
        types.clear();
        // batch size == 1 for vehicle type classify model
        // scan 1 by 1
        for (int i = 0; i < img_datas.size(); i++) {
            /* code */
            auto out = vehicleTypeModel->predictNoPadding({img_datas[i]}).front();
            auto it_max = std::max_element(out.begin(), out.end());
            auto id_max = it_max - out.begin();

            ObjCls objcls;
            objcls.class_ = id_max;
            objcls.score = *it_max;

            assert(types_.size() > id_max);
            objcls.label = types_[id_max];

            types.push_back(objcls);
        }
    }
}