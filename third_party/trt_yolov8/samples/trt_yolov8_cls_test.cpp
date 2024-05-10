

#include "../trt_yolov8_classifier.h"

int main() {
    trt_yolov8::trt_yolov8_classifier detector("./vp_data/models/trt/others/yolov8s-cls_v8.5.engine");
    
    auto image1 = cv::imread("./vp_data/test_images/vehicle_cls/1.jpg");
    auto image2 = cv::imread("./vp_data/test_images/vehicle_cls/5.jpg");
    std::unordered_map<int, std::string> labels_map;
    read_labels("./vp_data/models/imagenet_1000labels1.txt", labels_map);


    std::vector<std::vector<Classification>> classifications;
    std::vector<cv::Mat> images = {image1, image2};
    detector.classify(images, classifications, 5);  // top3 by default

    for (int i = 0; i < classifications.size(); ++i) {
        auto& classification = classifications[i];
        auto& image = images[i];

        for (int j = 0; j < classification.size(); ++j) {
            std::cout << "(top" << j + 1 << ") class_id:" << classification[j].class_id << "  conf:" << classification[j].conf << std::endl;
        }
        std::cout << std::endl;
        
        // draw top1's label on image
        cv::putText(image, "top1: " + labels_map.at(classification[0].class_id), cv::Point(10, 10), 1.5, 1, cv::Scalar(0, 0, 255));
    }

    cv::imshow("cls1", image1);
    cv::imshow("cls2", image2);

    cv::waitKey(0);
    return 0;
}