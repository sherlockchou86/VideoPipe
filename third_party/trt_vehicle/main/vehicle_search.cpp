#include <opencv2/core/core.hpp>
#include <opencv2/freetype.hpp>
#include <string>
#include <vector>
#include <memory>
#include <experimental/filesystem>

#include "../models/vehicle_feature_encoder.h"
#include "../../../utils/vp_utils.h"
#include "main.h"

using namespace std;

/*
*  vehicle search demo, 1:N
*/

#if VEHICLE_SEARCH

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

// extract feature for vehicle
void extract_feature(std::string vehicle_img_path, std::vector<float>& feature) {
    static std::string feature_model_path = "../data/model/vehicle/vehicle_embedding_v8.5.trt";
    static std::shared_ptr<trt_vehicle::VehicleFeatureEncoder> vehicleEncoder = nullptr;
    if (!vehicleEncoder) {
        vehicleEncoder = std::make_shared<trt_vehicle::VehicleFeatureEncoder>(feature_model_path);
    }

    auto vehicle = cv::imread(vehicle_img_path);
    std::vector<std::vector<float>> results;
    std::vector<cv::Mat> img_datas = {vehicle};
    vehicleEncoder->encode(img_datas, results);
    for(auto& i : results[0]) {
        feature.push_back(i);
    }
}

// load all vehicle images (.jpg) from disk and extract all features 
void load_vehicle_dataset(std::string dataset_dir, 
                        std::vector<std::pair<std::string, std::vector<float>>>& features_set) {
    features_set.clear();
    // iterate directory
    using recursive_directory_iterator = std::experimental::filesystem::recursive_directory_iterator;
    for (const auto& dir_entry : recursive_directory_iterator(dataset_dir))
        if (vp_utils::ends_with(dir_entry.path(), ".jpg")) {    
            std::cout << "load vehicle image: " << dir_entry << std::endl;

            // extract single feature
            std::vector<float> feature;
            extract_feature(dir_entry.path(), feature);

            std::pair<std::string, std::vector<float>> p {dir_entry.path(), feature};
            features_set.push_back(p);
        }
}

// match features using query feature, sorted by similiarity
void search(std::vector<float>& query_feature, 
            std::vector<std::pair<std::string, std::vector<float>>>& features_set, 
            std::vector<std::pair<std::string, float>>& query_result,
            int dis_type = 0,
            int top_n = 0) {
    query_result.clear();
    // just loop the features set
    for (auto& i: features_set) {
        /* code */
        auto dis = match(i.second, query_feature, dis_type);

        std::pair<std::string, float> p {i.first, dis};
        query_result.push_back(p);
    }
    
    // sort from high to low
    sort(query_result.begin(), query_result.end(), [=](std::pair<std::string, float>& a, std::pair<std::string, float>& b)
    {
        return a.second > b.second;
    });
}

int main() {
    auto vehicle_dataset_dir = "../data/test/vehicle_feature";
    auto query_vehicle_path = "../data/test/vehicle_feature/7_002.jpg";
    auto ft2 = cv::freetype::createFreeType2();
    ft2->loadFontData("../data/font/NotoSansCJKsc-Medium.otf", 0);

    // load vehicle dataset 
    std::vector<std::pair<std::string, std::vector<float>>> features_set;
    load_vehicle_dataset(vehicle_dataset_dir, features_set);

    // load query vehicle
    std::vector<float> query_feature;
    extract_feature(query_vehicle_path, query_feature);

    // search it!
    std::vector<std::pair<std::string, float>> query_result;
    search(query_feature, features_set, query_result, 0, 10);

    // print similiarity from high to low
    for(auto& i: query_result) {
        std::cout << i.second << " ==> " << i.first << std::endl;
    }

    // display according to query_result
    auto n_query = query_result.size();
    auto rect_w_h = 80; auto gap = 20; auto cols = 10;
    auto rows = n_query / cols + 2;
    
    // create canvas
    cv::Mat canvas(rows * (rect_w_h + gap) + gap, cols * (rect_w_h + gap) + gap, CV_8UC3,cv::Scalar(127, 127, 127));
    
    // query vehicle at first line
    cv::Mat roi_query = cv::Mat(canvas, cv::Rect(gap, gap, rect_w_h, rect_w_h));
    auto query_vehicle_img = cv::imread(query_vehicle_path);
    cv::Mat query_vehicle_img_tmp;
    cv::resize(query_vehicle_img, query_vehicle_img_tmp, cv::Size(rect_w_h, rect_w_h));
    query_vehicle_img_tmp.copyTo(roi_query);
    ft2->putText(canvas, "query vehicle:", cv::Point(20, 14), 14, cv::Scalar(255, 0, 0), cv::FILLED, cv::LINE_AA, true);

    // query result
    for(int i = 0; i < query_result.size(); ++i) {
        auto row = i / cols + 1;
        auto col = i % cols;
        cv::Mat roi = cv::Mat(canvas, cv::Rect(gap + col * (rect_w_h + gap), gap + row * (rect_w_h + gap), rect_w_h, rect_w_h));
        auto vehicle_img = cv::imread(query_result[i].first);
        cv::Mat vehicle_img_tmp;
        cv::resize(vehicle_img, vehicle_img_tmp, cv::Size(rect_w_h, rect_w_h));
        vehicle_img_tmp.copyTo(roi);

        ft2->putText(canvas, std::to_string(std::max(query_result[i].second, 0.0f)), cv::Point(gap + col * (rect_w_h + gap), row * (rect_w_h + gap) + gap), 14, cv::Scalar(255, 0, 0), cv::FILLED, cv::LINE_AA, true);
    }

    cv::imshow("search result", canvas);
    cv::waitKey(0);
    return 0;
}

#endif
