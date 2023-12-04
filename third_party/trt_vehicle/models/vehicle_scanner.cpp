#include "vehicle_scanner.h"


namespace trt_vehicle {
    VehicleScanner::VehicleScanner(const std::string& scanModelPath, float scoreThres, float iouThres) {
        scannerModel = new DetectModel(scanModelPath, scoreThres, iouThres);
    }
    
    VehicleScanner::~VehicleScanner() {
        if (scannerModel) {
            delete scannerModel;
        }
    }
    
    void VehicleScanner::detect(std::vector<cv::Mat>& img_datas, std::vector<std::vector<trt_vehicle::ObjBox>>& parts) {
        parts.clear();
        // batch size == 1 for vehicle det model
        // scan 1 by 1
        for (int i = 0; i < img_datas.size(); i++) {
            std::vector<std::vector<ObjBox>> outBoxes;
            scannerModel->predictPadding({img_datas[i]}, outBoxes, 128);

            auto& outBox = outBoxes[0];
            std::vector<trt_vehicle::ObjBox> part_list;

            if (outBox.size() == 0) {
                parts.push_back(part_list);
                continue;
            }

            for (auto box: outBox) {
                // attach vehicle label field
                box.label = partLabels[box.class_];
                part_list.push_back(box);
            }
            
            parts.push_back(part_list);
        }
    } 
}