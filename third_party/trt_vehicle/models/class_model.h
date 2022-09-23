#pragma once

#include "base_model.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace trt_vehicle{
    class ClassModel: public BaseModel{
    public:
        //init class model
        ClassModel(const std::string& modelPath, bool isVehicle= true, std::string meanfile="");
        ~ClassModel();

        //inferential prediction of the picture no pad
        std::vector<std::vector<float>> predictNoPadding(std::vector<cv::Mat> imgs);
        //inferential prediction of the picture after pad
        std::vector<std::vector<float>> predictPadding(std::vector<cv::Mat> imgs,int paddingValue);
        
    private:
        int setSize();
        int extractFeature(std::vector<cv::Mat> imgs);
    private:
        int m_outputSizeC;

        float* m_scoreCuda;
        float* m_score;
        float* m_imageDataCuda;

        CudaPredictor m_predictor;

		bool isVehicleModel = false;
    };
}