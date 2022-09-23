#include <opencv2/core/core.hpp>
#include <string>
#include <vector>
#include <map>

#include "../models/vehicle_plate_detector.h"
#include "../models/vehicle_detector.h"
#include "main.h"

using namespace std;

/*
* vehicle detect demo
*/

#if VEHICLE

std::map<std::string, cv::Scalar> colors {{"car", cv::Scalar(255, 0, 0)},
                                        {"bus", cv::Scalar(0, 255, 255)},
                                        {"truck", cv::Scalar(0, 255, 0)}};

int main() {
    std::string vehicle_model_path = "../data/model/vehicle/vehicle.trt";
    auto image = cv::imread("../data/test/vehicle/4.png");
    auto image2 = cv::imread("../data/test/vehicle/3.jpg");

    auto vehicle_detector = new trt_vehicle::VehicleDetector(vehicle_model_path);

    std::vector<cv::Mat> img_datas = {image, image2};
    std::vector<std::vector<trt_vehicle::ObjBox>> results;

    vehicle_detector->detect(img_datas, results);

    for (int i = 0; i < results.size(); i++)
    {
        /* code */
        for (int j = 0; j < results[i].size(); j++)
        {
            /* code */
            auto& objbox = results[i][j];

            cv::rectangle(img_datas[i], cv::Rect(objbox.x, objbox.y, objbox.width, objbox.height), colors.at(objbox.label), 2);
            cv::putText(img_datas[i], objbox.label + std::to_string(objbox.score), cv::Point(objbox.x, objbox.y), 1, 0.8, colors.at(objbox.label));
        }
        
    }
    cv::imshow("image", image);
    cv::imshow("image2", image2);
    
    cv::waitKey(0);
    delete vehicle_detector;
}
#endif