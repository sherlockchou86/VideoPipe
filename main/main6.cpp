

#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "VP.h"

#if MAIN6




int main() {
    std::vector<std::string> labels;
    try {
        std::ifstream label_stream("./models/imagenet_1000labels1.txt");
        for (std::string line; std::getline(label_stream, line); ) {
            labels.push_back(line);
        }
    }
    catch(const std::exception& e) {
        
    }  


    auto net = cv::dnn::readNet("./models/resnet18-v1-7.onnx","");
    auto image = cv::imread("./10.png");
    auto image2 = cv::imread("./9.jpg");
    std::vector<float> std_vec {0.229, 0.224, 0.225};

    cv::Mat blob = cv::dnn::blobFromImages(std::vector<cv::Mat>{image, image2}, 1/255.0, cv::Size(128, 128), cv::Scalar(123.675, 116.28, 103.53), true);
/*
    for (size_t i = 0; i < blob.size[0]; i++)
    {
        for (size_t j = 0; j < blob.size[1]; j++)
        {
            for (size_t k = 0; k < blob.size[2]; k++)
            {
                for (size_t l = 0; l < blob.size[3]; l++)
                {
                    int vec[4] = {i,j,k,l};
                    blob.at<float>(vec) /= std_vec[j];
                }
                
            }
            
        }
        
    }*/
    

    std::cout << blob.size.dims() << " " << blob.size[0] << " " << blob.size[1] << " " << blob.size[2] << " " << blob.size[3] << std::endl;

    net.setInput(blob);
    cv::Mat output;
    output = net.forward();
    
    cv::Point classIdPoint;
    double confidence;


    float maxProb = 0.0;
    float sum = 0.0;
    cv::Mat softmaxProb;
    auto r1 = output.row(0);

    maxProb = *std::max_element(r1.begin<float>(), r1.end<float>());
    cv::exp(r1-maxProb, softmaxProb);
    sum = (float)cv::sum(softmaxProb)[0];
    softmaxProb /= sum;
    minMaxLoc(softmaxProb.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
    int classId = classIdPoint.x;
    std::cout << classId << "---" << confidence << "---" << labels[classId] << std::endl;


    auto r2 = output.row(1);
    maxProb = *std::max_element(r2.begin<float>(), r2.end<float>());
    cv::exp(r2-maxProb, softmaxProb);
    sum = (float)cv::sum(softmaxProb)[0];
    softmaxProb /= sum;
    minMaxLoc(softmaxProb.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
    classId = classIdPoint.x;
    std::cout << classId << "---" << confidence << "---" << labels[classId] << std::endl;
}

#endif