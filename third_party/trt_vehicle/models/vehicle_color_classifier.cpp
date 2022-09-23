
#include "vehicle_color_classifier.h"

namespace trt_vehicle {
    
    VehicleColorClassifier::VehicleColorClassifier(const std::string& vehicleColorModelPath) {
        vehicleColorModel = new ClassModel(vehicleColorModelPath);
    }
    
    VehicleColorClassifier::~VehicleColorClassifier() {
        if (vehicleColorModel != nullptr) {
            delete vehicleColorModel;
        }
    }

    void VehicleColorClassifier::classify(std::vector<cv::Mat>& img_datas, std::vector<trt_vehicle::ObjCls>& colors) {
        colors.clear();
        // batch size == 1 for vehicle color classify model
        // scan 1 by 1
        for (int i = 0; i < img_datas.size(); i++) {
            /* code */
            auto out = vehicleColorModel->predictNoPadding({img_datas[i]}).front();
            auto it_max = std::max_element(out.begin(), out.end());
            auto id_max = it_max - out.begin();

            ObjCls objcls;
            objcls.class_ = id_max;
            objcls.score = *it_max;

            assert(colors_.size() > id_max);
            objcls.label = colors_[id_max];

            colors.push_back(objcls);
        }
    }
}