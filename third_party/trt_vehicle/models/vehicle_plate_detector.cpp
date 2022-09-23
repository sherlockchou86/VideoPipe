#include "vehicle_plate_detector.h"

using namespace std;

namespace trt_vehicle{
    VehiclePlateDetector::VehiclePlateDetector(const std::string& plateModelPath,
                                const std::string& charModelPath,
                                float plateScoreThres,
                                float plateIouThres,
                                float charScoreThres,
                                float charIouThres) {
        plateModel = new DetectModel(plateModelPath, plateScoreThres, plateIouThres);
        charModel = new DetectModel(charModelPath, charScoreThres, charIouThres);
    }

    VehiclePlateDetector::~VehiclePlateDetector() {
        if(plateModel) {
            delete plateModel;
        }
        if(charModel) {
            delete charModel;
        }
    }
    
    void VehiclePlateDetector::detect(std::vector<cv::Mat>& img_datas, std::vector<std::vector<Plate>>& plates, bool only_one) {
        plates.clear();
        // batch size == 1 for plate det and char rec model
        // scan 1 by 1
        for (int i = 0; i < img_datas.size(); i++) {
            std::vector<std::vector<ObjBox>> outBoxes;
            // detect
            plateModel->predictPadding({img_datas[i]}, outBoxes, 128);
            std::vector<ObjBox>& outBox = outBoxes[0];
            std::vector<Plate> plate_list;
            if (outBox.size() == 0) {
                plates.push_back(plate_list);
                continue;
            }

            // crop
            std::vector<cv::Mat> imgSegs;
            auto crop_func = [&](ObjBox& box) {
                float saclePara = 0.3;
                int bia = saclePara * box.height;
                int x2 = int(box.width+box.x + bia);
                int y2 = int(box.height+box.y + bia);
                box.x = max(1, int(box.x-bia));
                box.y = max(1, int(box.y-bia));
                box.width = min(img_datas[i].size().width-1, x2) - box.x;
                box.height = min(img_datas[i].size().height-1, y2) - box.y;
                cv::Rect rect = cv::Rect(int(box.x), int(box.y), int(box.width), int(box.height));
                cv::Mat imgSeg = img_datas[i](rect);
                imgSegs.push_back(imgSeg);
            };
            
            // only one plate for each image
            int only_one_index = -1;
            if (only_one) {
                std::vector<float> scores;
                for(auto& out : outBox){
                    scores.push_back(out.score);
                }
                std::vector<int> indices = getSortIndex(scores);
                auto& box = outBox[indices[0]];
                only_one_index = indices[0];
                crop_func(box);
            }
            else {
                for (int k = 0; k < outBox.size(); k++) {
                    auto& box = outBox[k];
                    crop_func(box);
                }
            }

            // recognition
            for (int m = 0; m < imgSegs.size(); m++) {
                std::vector<std::vector<ObjBox>> charOutBoxes;
                charModel->predictPadding({imgSegs[m]}, charOutBoxes, 128);
                auto& charOutBox = charOutBoxes[0];

                Plate plate;
                std::string plate_text = "";
                auto the_box_index = only_one ? only_one_index : m;

                std::vector<float> Xs;
                for (auto& out : charOutBox) {
                    out.x = out.x + outBox[the_box_index].x;
                    out.y = out.y + outBox[the_box_index].y;
                    Xs.push_back(-1.0 * out.x);
                }

                std::vector<int> indices = getSortIndex(Xs);
                for(int j = 0; j < indices.size(); j++) {
                    plate_text = plate_text + chars[charOutBox[indices[j]].class_];
                }
                plate.x = outBox[the_box_index].x;
                plate.y = outBox[the_box_index].y;
                plate.width = outBox[the_box_index].width;
                plate.height = outBox[the_box_index].height;
                plate.text = plate_text;
                plate.color = colors[outBox[the_box_index].class_];

                plate_list.push_back(plate);
            }

            plates.push_back(plate_list);
        }
    }
}
