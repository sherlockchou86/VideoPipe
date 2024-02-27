#include <opencv2/core/core.hpp>
#include <opencv2/freetype.hpp>
#include <string>
#include <vector>
#include <memory>
#include <experimental/filesystem>

#include "../models/vehicle_feature_encoder.h"
#include "../../../utils/vp_utils.h"
#include "../../bhtsne/tsne.h"  // t-SNE algo

using namespace std;

/*
*  vehicle cluster demo, reduce dims of features and display them on 2D screen.
*/

// extract feature for vehicle
void extract_feature(std::string vehicle_img_path, std::vector<float>& feature) {
    static std::string feature_model_path = "./vp_data/models/trt/vehicle/vehicle_embedding_v8.5.trt";
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


void reduce_dims(std::vector<std::pair<std::string, std::vector<float>>>& features_set,
                 std::vector<std::pair<std::string, std::vector<float>>>& low_dims_features_set,
                /* default parameters for t-SNE algorithm */
                int no_dims = 2, int max_iter = 1000, double perplexity = 2, double theta = 0.5, int rand_seed = -1, bool skip_random_init = false, int stop_lying_iter = 250, int mom_switch_iter = 250) {
    assert(features_set.size() != 0);
    auto N = features_set.size();
    auto D = features_set[0].second.size();  // all the same as the first feature's dims

    // prepare input
    double data[N * D];
    double Y[N * no_dims];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            data[i * D + j] = features_set[i].second[j];
        }
    }
    
    // call t-SNE
    TSNE::run(data, N, D, Y, no_dims, perplexity, theta, rand_seed, skip_random_init, max_iter, stop_lying_iter, mom_switch_iter);

    // prepare output
    for (int i = 0; i < N; i++) {
        std::vector<float> low_dims_feature;
        for (int j = 0; j < no_dims; j++) {
            low_dims_feature.push_back(float(Y[i * no_dims + j]));
        }

        std::pair<std::string, std::vector<float>> p {features_set[i].first, low_dims_feature};
        low_dims_features_set.push_back(p);
    }
}

int main() {
    auto vehicle_dataset_dir = "./vp_data/test_images/vehicle_feature";
    auto ft2 = cv::freetype::createFreeType2();
    ft2->loadFontData("./vp_data/font/NotoSansCJKsc-Medium.otf", 0);

    // load vehicle dataset 
    std::vector<std::pair<std::string, std::vector<float>>> features_set;
    load_vehicle_dataset(vehicle_dataset_dir, features_set);

    // reduce dims using t-SNE
    std::vector<std::pair<std::string, std::vector<float>>> low_dims_feature_set;
    reduce_dims(features_set, low_dims_feature_set);

    // print low dims features
    std::cout << "low dims features:" << std::endl;
    for(auto& i: low_dims_feature_set) {
        std::cout << "[" << std::endl;
        for(auto& j: i.second) {
            std::cout << j << ",";
        }
        std::cout << "]" << std::endl;
    }

    // normalize low dims feature to coordinate of [0:1] and display them on 2D screen
    auto max_x = 0.0f, max_y = 0.0f, min_x = 0.0f, min_y = 0.0f;
    for(auto& i: low_dims_feature_set) {
        auto& f = i.second;
        // 2 values in f
        max_x = std::max(max_x, f[0]);
        max_y = std::max(max_y, f[1]);
        min_x = std::min(min_x, f[0]);
        min_y = std::min(min_y, f[1]);
    }
    auto x_range = max_x - min_x;
    auto y_range = max_y - min_y;

    // draw on (canvas_w_h + img_w_h) * (canvas_w_h + img_w_h)
    auto canvas_w_h = 800, img_w_h = 70;
    cv::Mat canvas(canvas_w_h + img_w_h, canvas_w_h + img_w_h, CV_8UC3, cv::Scalar(127, 127, 127));
    for(auto& i: low_dims_feature_set) {
        auto& f =i.second;
        // convert to [0:1]
        f[0] = (f[0] - min_x) / x_range;
        f[1] = (f[1] - min_y) / y_range;

        auto img = cv::imread(i.first);
        cv::Mat img_tmp;
        cv::resize(img, img_tmp, cv::Size(img_w_h, img_w_h));
        cv::rectangle(img_tmp, cv::Rect(0, 0, img_tmp.cols, img_tmp.rows), cv::Scalar(255, 0, 0));
        cv::Mat roi(canvas, cv::Rect(int(f[0] * canvas_w_h), int(f[1] * canvas_w_h), img_w_h, img_w_h));
        img_tmp.copyTo(roi);
    }

    cv::imshow("cluster result", canvas);
    cv::waitKey(0);
    return 0;
}