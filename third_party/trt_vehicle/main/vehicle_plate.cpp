#include <opencv2/core/core.hpp>
#include <string>
#include <vector>

#include "../models/vehicle_plate_detector.h"
#include "main.h"

using namespace std;

/*
*  vehicle plate demo
*/

#if VEHICLE_PLATE

std::map<std::string, cv::Scalar> colors {{"blue", cv::Scalar(255, 0, 0)},
                                        {"yellow", cv::Scalar(0, 255, 255)},
                                        {"green", cv::Scalar(0, 255, 0)},
                                        {"white", cv::Scalar(255, 255, 255)}};

int main(){
    std::string plateModelPath = "../data/model/plate/det.trt";
    std::string charModelPath = "../data/model/plate/rec.trt";
    auto image = cv::imread("../data/test/plate/truck2.png");
    auto image2 = cv::imread("../data/test/plate/3in1.png");
    auto plateDetector = new trt_vehicle::VehiclePlateDetector(plateModelPath, charModelPath);

    std::vector<std::vector<trt_vehicle::Plate>> results;
    std::vector<cv::Mat> img_datas = {image, image2};
    plateDetector->detect(img_datas, results, true);

    for (int i = 0; i < results.size(); i++) {
        /* code */
        for (int j = 0; j < results[i].size(); j++)
        {
            /* code */
            auto& plate = results[i][j];
            cv::rectangle(img_datas[i], cv::Rect(plate.x, plate.y, plate.width, plate.height), colors.at(plate.color));
            cv::putText(img_datas[i], plate.text, cv::Point(plate.x, plate.y), 1, 0.8, colors.at(plate.color));
        }
        
    }
    cv::imshow("image", image);
    cv::imshow("image2", image2);
    
    cv::waitKey(0);
    delete plateDetector;
    return 0;
}

#endif
