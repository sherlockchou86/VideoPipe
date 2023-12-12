#include <opencv2/core/core.hpp>
#include <opencv2/freetype.hpp>
#include <string>
#include <vector>

#include "../models/vehicle_feature_encoder.h"
#include "main.h"

using namespace std;

/*
*  vehicle comparision demo, 1:1
*/

#if VEHICLE_COMPARISION

// compare two features
// dis_type 0 means cosine distance, bigger means more similiar
// dis_type 1 means L2 distance, smaller means more similiar
double match(std::vector<float>& feature1, std::vector<float>& feature2, int dis_type) {
    auto _face_feature1 = cv::Mat(1, feature1.size(), CV_32F, feature1.data());
    auto _face_feature2 = cv::Mat(1, feature2.size(), CV_32F, feature2.data());
    cv::normalize(_face_feature1, _face_feature1);
    cv::normalize(_face_feature2, _face_feature2);

    if(dis_type == 0) {
        return cv::sum(_face_feature1.mul(_face_feature2))[0];
    }
    else if(dis_type == 1) {
        return cv::norm(_face_feature1, _face_feature2);
    }
    else {
        return 0;
    }
}

int main() {
    std::string featureModelPath = "../data/model/vehicle/vehicle_embedding_v8.5.trt";
    auto image = cv::imread("../data/test/vehicle_feature/2_001.jpg");  // first vehicle
    auto image2 = cv::imread("../data/test/vehicle_feature/2_003.jpg");  // second vehicle to compare(the same vehicle with different angle)
    auto image3 = cv::imread("../data/test/vehicle_feature/5_002.jpg");  // third vehicle to compare(not the same vehicle)
    auto vehicleEncoder = new trt_vehicle::VehicleFeatureEncoder(featureModelPath);

    auto ft2 = cv::freetype::createFreeType2();
    ft2->loadFontData("../data/font/NotoSansCJKsc-Medium.otf", 0);

    std::vector<std::vector<float>> results;
    std::vector<cv::Mat> img_datas = {image, image2, image3};
    vehicleEncoder->encode(img_datas, results); // encode

    auto feature1 = results[0];
    auto feature2 = results[1];
    auto feature3 = results[2];

    // print features, 256 dims
    auto print_features = [](std::vector<float> features, std::string name) {
        std::cout << name << ": " << features.size() << " dims " << "[" << std::endl;
        for (auto& i: features) {
            std::cout << i << ",";
        }
        std::cout << "]" << std::endl;
    };
    print_features(feature1, "feature1");
    print_features(feature2, "feature2");
    print_features(feature3, "feature3");

    auto dis_type = 0;
    auto dis_1_2 = match(feature1, feature2, dis_type);  // compare feature1 and feature2
    auto dis_1_3 = match(feature1, feature3, dis_type);  // compare feature1 and feature3
    std::cout << "distance between feature1 and feature2: " << dis_1_2 << std::endl;
    std::cout << "distance between feature1 and feature3: " << dis_1_3 << std::endl;

    // convert to similarity
    auto similiarity_1_2 = dis_type == 0 ? dis_1_2 : (1 - dis_1_2);
    similiarity_1_2 = std::max(similiarity_1_2, 0.0);
    auto similiarity_1_3 = dis_type == 0 ? dis_1_3 : (1 - dis_1_3);
    similiarity_1_3 = std::max(similiarity_1_3, 0.0);

    // display
    auto h = image.rows + std::max(image2.rows, image3.rows) + 50 + 10 + 10;
    auto w = std::max(image2.cols, image3.cols) * 2 + 50 + 10 + 10;

    cv::Mat canvas(h, w, CV_8UC3,cv::Scalar(127, 127, 127));
    cv::Mat roi_(canvas,cv::Rect(canvas.cols / 2 - image.cols / 2, 10, image.cols, image.rows));
    cv::Mat roi_2(canvas, cv::Rect(canvas.cols / 2 - image2.cols - 25, 10 + image.rows + 25, image2.cols, image2.rows));
    cv::Mat roi_3(canvas, cv::Rect(canvas.cols / 2 + 25, 10 + image.rows + 25, image3.cols, image3.rows));

    image.copyTo(roi_);
    image2.copyTo(roi_2);
    image3.copyTo(roi_3);

    cv::line(canvas, cv::Point(canvas.cols / 2, 10 + image.rows), cv::Point(10 + image2.cols / 2, 10 + image.rows + 50), cv::Scalar(255, 0, 0));
    cv::line(canvas, cv::Point(canvas.cols / 2, 10 + image.rows), cv::Point(canvas.cols / 2 + 25 + image3.cols / 2, 10 + image.rows + 50), cv::Scalar(255, 0, 0));
    ft2->putText(canvas, "相似度：" + std::to_string(similiarity_1_2), cv::Point(10, 10 + image.rows + 25), 20, cv::Scalar(255, 0, 0), cv::FILLED, cv::LINE_AA, true);
    ft2->putText(canvas, "相似度：" + std::to_string(similiarity_1_3), cv::Point(10 + image2.cols + 50, 10 + image.rows + 25), 20, cv::Scalar(255, 0, 0), cv::FILLED, cv::LINE_AA, true);

    cv::imshow("compare result", canvas);
    cv::waitKey(0);
    delete vehicleEncoder;
    return 0;
}

#endif
