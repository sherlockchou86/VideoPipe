#include <opencv2/core/core.hpp>
#include <opencv2/freetype.hpp>
#include <string>
#include <vector>
#include <map>

#include "../models/vehicle_color_classifier.h"
#include "../models/vehicle_type_classifier.h"

using namespace std;

/*
* vehicle color and vehicle type classifier demo
* tensorRT 8.5 + CUDA 11.1 + cuDNN 8.6 for this code, because original onnx weight could not be converted to TRT engine on tensorRT 7.2
*    
* ## color ##
*    black
*    blue
*    grey
*    other
*    red
*    white
*    yellow
*
* ## type ##
*    bus
*    business_car
*    construction_truck
*    large_truck
*    sedan
*    small_truck
*    suv
*    tanker
*    van
*/

// to Chinese
std::map<std::string, std::string> types {{"bus", "巴士"},
                                        {"business_car", "商务面包车"},
                                        {"construction_truck", "施工车"},
                                        {"large_truck", "大卡车"},
                                        {"sedan", "轿车"},
                                        {"small_truck", "小卡车"},
                                        {"suv", "SUV"},
                                        {"tanker", "罐车"},
                                        {"van", "厢式货车"}};

int main() {
    std::string vehicle_color_model_path = "./vp_data/models/trt/vehicle/vehicle_color_v8.5.trt";
    std::string vehicle_type_model_path = "./vp_data/models/trt/vehicle/vehicle_type_v8.5.trt";
    std::string font_path = "./vp_data/font/NotoSansCJKsc-Medium.otf";

    auto image = cv::imread("./vp_data/test_images/vehicle_cls/1.jpg");     // vehicle in the center of image, crop it first if possible
    auto image2 = cv::imread("./vp_data/test_images/vehicle_cls/2.jpg");
    auto image3 = cv::imread("./vp_data/test_images/vehicle_cls/3.jpg");
    auto image4 = cv::imread("./vp_data/test_images/vehicle_cls/4.jpg");
    auto image5 = cv::imread("./vp_data/test_images/vehicle_cls/5.jpg");
    auto image6 = cv::imread("./vp_data/test_images/vehicle_cls/6.jpg");

    cv::Ptr<cv::freetype::FreeType2> ft2 = cv::freetype::createFreeType2();
    ft2->loadFontData(font_path, 0); 

    auto vehicle_color_classifier = new trt_vehicle::VehicleColorClassifier(vehicle_color_model_path);
    auto vehicle_type_classifier = new trt_vehicle::VehicleTypeClassifier(vehicle_type_model_path);
    std::vector<cv::Mat> img_datas = {image, image2, image3, image4, image5, image6};
    std::vector<trt_vehicle::ObjCls> color_results;
    std::vector<trt_vehicle::ObjCls> type_results;

    vehicle_color_classifier->classify(img_datas, color_results);  // color classify
    vehicle_type_classifier->classify(img_datas, type_results);   // type classify

    for (int i = 0; i < img_datas.size(); i++) {
        /* draw class label at top left */
        // cv::putText(img_datas[i], color_results[i].label + " " + types.at(type_results[i].label), cv::Point(30, 30), 1, 2, colors.at(color_results[i].label));
        ft2->putText(img_datas[i], color_results[i].label + " " + types.at(type_results[i].label), cv::Point(30, 30), 25, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_AA, true);
    }
    
    cv::imshow("image", image);
    cv::imshow("image2", image2);
    cv::imshow("image3", image3);
    cv::imshow("image4", image4);
    cv::imshow("image5", image5);
    cv::imshow("image6", image6);
    
    cv::waitKey(0);
    delete vehicle_color_classifier;
    delete vehicle_type_classifier;
}