#pragma once

#include "base_model.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>


namespace trt_vehicle{
    class DetectModel: public BaseModel{
    public:
        DetectModel(const std::string& modelPath,float scoreThres,float iouThres);
        ~DetectModel();

        //inferential prediction of the picture after pad
        int predictPadding(std::vector<cv::Mat> imgs,std::vector<std::vector<ObjBox>>& outBoxes, int paddingValue);

    private:
        int predict(std::vector<cv::Mat> imgs,std::vector<std::vector<ObjBox>>& outBoxes);
        int setSize();
        int extractFeature(std::vector<cv::Mat> imgs);
        
    private:
        int m_outputNum;
        int m_classNum;
        int m_boxNum;
        float m_scoreThres;
        float m_iouThres;
        float m_ratioH;
        float m_ratioW;

        std::vector<int> m_hStarts;
        std::vector<int> m_wStarts;

        float* m_scoreCuda;
        float* m_boxCuda;
        float* m_score;
        float* m_box;
        float* m_imageDataCuda;

        CudaPredictor m_predictor;
    };
}
