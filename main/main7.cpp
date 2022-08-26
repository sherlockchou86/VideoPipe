#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "VP.h"

#if MAIN7

int main() {
    std::vector<std::string> labels;
    try {
        std::ifstream label_stream("./models/yolov3_5classes.txt");
        for (std::string line; std::getline(label_stream, line); ) {
            labels.push_back(line);
        }
    }
    catch(const std::exception& e) {
        
    }  


    auto net = cv::dnn::readNet("./models/yolov3-5_2022-0415_best.weights", "./models/yolov3-5_2022-0415.cfg");
    auto capture = cv::VideoCapture("./3.mp4");
    while(1) {
        cv::Mat image, image2;
        capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        if(!capture.read(image)) {
            break;
        }
        if (!capture.read(image2)) {
            break;
        }
        cv::resize(image, image, cv::Size(1280, 720));
        cv::resize(image2, image2, cv::Size(1280, 720));

        std::vector<float> std_vec {0.229, 0.224, 0.225};
        std::vector<cv::Mat> images {image, image2};
        cv::Mat blob = cv::dnn::blobFromImages(images, 1 / 255.0, cv::Size(416, 416), cv::Scalar(0), true, false, CV_32F);
        std::cout << blob.size.dims() << " " << blob.size[0] << " " << blob.size[1] << " " << blob.size[2] << " " << blob.size[3] << std::endl;

        net.setInput(blob);
        std::vector<cv::Mat> raw_outputs;
        net.forward(raw_outputs, net.getUnconnectedOutLayersNames());

        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        auto score_threshold = 0.1;
        auto confidence_threshold = 0.1;
        auto nms_threshold = 0.5;

        auto batch = raw_outputs[0].size[0];

        for (int k = 0; k < batch; k++)  // batch size
        {
            for (int i = 0; i < raw_outputs.size(); ++i) {  // output size
                cv::Mat output = cv::Mat(raw_outputs[i].size[1], raw_outputs[i].size[2], CV_32F, raw_outputs[i].ptr(k));
                std::cout << output << std::endl;

                auto data = (float*)output.data;
                for (int j = 0; j < output.rows; ++j, data += output.cols) {
                    
                    float confidence = data[4];
                    // check confidence threshold
                    if (confidence < confidence_threshold) {
                        continue;
                    }

                    cv::Mat scores = output.row(j).colRange(5, output.cols);
                    cv::Point class_id;
                    double max_score;
                    // Get the value and location of the maximum score
                    cv::minMaxLoc(scores, 0, &max_score, 0, &class_id);

                    // check score threshold
                    if (max_score >= score_threshold) {
                        int centerX = (int)(data[0] * image.cols);
                        int centerY = (int)(data[1] * image.rows);
                        int width = (int)(data[2] * image.cols);
                        int height = (int)(data[3] * image.rows);
                        int left = centerX - width / 2;
                        int top = centerY - height / 2;

                        class_ids.push_back(class_id.x);
                        confidences.push_back((float)confidence);
                        boxes.push_back(cv::Rect(left, top, width, height));
                    }
                }
            }

            std::vector<int> indices;
            cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold, indices);

            // create target
            for (int i = 0; i < indices.size(); ++i) {
                int idx = indices[i];
                auto box = boxes[idx];
                auto confidence = confidences[idx];
                auto class_id = class_ids[idx];
                auto label = (labels.size() < class_id + 1) ? "" : labels[class_id];

                cv::putText(images[k], label, cv::Point(box.x, box.y), 1, 1, cv::Scalar(255, 255, 0));
                cv::rectangle(images[k], box, cv::Scalar(255, 0, 0), 2);
            }

            class_ids.clear();
            confidences.clear();
            boxes.clear();
        }


        cv::imshow("detect", images[0]);
        cv::imshow("detect2", images[1]);
        cv::waitKey(1);
    }
}

#endif