#include <opencv2/core/core.hpp>
#include <string>
#include <vector>
#include <map>

#include "../models/vehicle_scanner.h"
#include "main.h"

using namespace std;

/*
* vehicle scan demo, detect on vehicle body
* tensorRT 8.5 + CUDA 11.1 + cuDNN 8.6 for this code, because original onnx weight could not be converted to TRT engine on tensorRT 7.2
*/

#if VEHICLE_SCAN

std::map<std::string, cv::Scalar> colors {{"truck", cv::Scalar(255, 0, 0)},  // 货车
                                        {"bus", cv::Scalar(0, 255, 255)},  //巴士
                                        {"car", cv::Scalar(0, 255, 0)},  //轿车
                                        {"suv", cv::Scalar(0, 255, 0)}, //suv+商务
                                        {"van", cv::Scalar(0, 255, 0)}, //面包+金杯
                                        {"nowindow", cv::Scalar(0, 255, 0)},  //封窗货车，货拉拉
                                        {"iveco", cv::Scalar(0, 255, 0)},  //9座依维柯
                                        {"pickup", cv::Scalar(0, 255, 0)},  //皮卡
                                        {"wheel", cv::Scalar(0, 255, 0)}};  //轮子

int main() {
    std::string vehicle_scan_model_path = "../data/model/vehicle/vehicle_scan_v8.5.trt";
    auto image = cv::imread("../data/test/vehicle_body/0.jpg");
    auto image2 = cv::imread("../data/test/vehicle_body/1.jpg");

    auto vehicle_scanner = new trt_vehicle::VehicleScanner(vehicle_scan_model_path);

    std::vector<cv::Mat> img_datas = {image, image2};
    std::vector<std::vector<trt_vehicle::ObjBox>> results;

    vehicle_scanner->detect(img_datas, results);

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
    delete vehicle_scanner;
}
#endif