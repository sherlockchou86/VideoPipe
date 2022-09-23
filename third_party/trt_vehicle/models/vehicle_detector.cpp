

#include "vehicle_detector.h"


namespace trt_vehicle {
        
    VehicleDetector::VehicleDetector(const std::string& vehicleModelPath, 
                                    float vehicleScoreThres, 
                                    float vehicleIouThres) {
        vehicleModel = new DetectModel(vehicleModelPath, vehicleScoreThres, vehicleIouThres);
    }
    
    VehicleDetector::~VehicleDetector() {
        if (vehicleModel) {
            delete vehicleModel;
        }
    }
    
    void VehicleDetector::detect(std::vector<cv::Mat>& img_datas, std::vector<std::vector<trt_vehicle::ObjBox>>& vehicles) {
        vehicles.clear();
        // batch size == 1 for vehicle det model
        // scan 1 by 1
        for (int i = 0; i < img_datas.size(); i++) {
            std::vector<std::vector<ObjBox>> outBoxes;
            vehicleModel->predictPadding({img_datas[i]}, outBoxes, 128);

            auto& outBox = outBoxes[0];
            std::vector<trt_vehicle::ObjBox> vehicle_list;

            if (outBox.size() == 0) {
                vehicles.push_back(vehicle_list);
                continue;
            }

            for (auto box: outBox) {
                // attach vehicle label field
                box.label = vehicle_labels[box.class_];
                vehicle_list.push_back(box);
            }
            
            vehicles.push_back(vehicle_list);
        }
    }
}