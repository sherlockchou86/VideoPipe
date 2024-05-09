#pragma once

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "include/cuda_utils.h"
#include "include/logging.h"
#include "include/model.h"
#include "include/postprocess.h"
#include "include/preprocess.h"
#include "include/utils.h"

namespace trt_yolov8 {
    using namespace nvinfer1;
    class trt_yolov8_pose_detector
    {
    private:
        /* data */

        void serialize_engine(std::string& wts_name, std::string& engine_name, int& is_p, std::string& sub_type, float& gd,
                            float& gw, int& max_channels);
        void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine,
                                IExecutionContext** context);
        void prepare_buffer(ICudaEngine* engine, float** input_buffer_device, float** output_buffer_device,
                            float** output_buffer_host, float** decode_ptr_host, float** decode_ptr_device,
                            std::string cuda_post_process);
        void infer(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output, int batchsize,
                float* decode_ptr_host, float* decode_ptr_device, int model_bboxes, std::string cuda_post_process);

        Logger gLogger;
        const int kOutputSize = kMaxNumOutputBbox * (sizeof(Detection) - sizeof(float) * 32) / sizeof(float) + 1;
        cudaStream_t stream;
        int model_bboxes;

        // Deserialize the engine from file
        nvinfer1::IRuntime* runtime = nullptr;
        nvinfer1::ICudaEngine* engine = nullptr;
        nvinfer1::IExecutionContext* context = nullptr;
        
        std::string cuda_post_process = "c";
    public:
        trt_yolov8_pose_detector(std::string model_path = "");
        ~trt_yolov8_pose_detector();

        // detect
        void detect(std::vector<cv::Mat> images, std::vector<std::vector<Detection>>& detections);
        
        // serialize wts to plan file for pose estimate
        // sub_type: [ n/s/m/l/x/n2/s2/m2/l2/x2/n6/s6/m6/l6/x6 ]
        bool wts_2_engine(std::string wts_name, std::string engine_name, std::string sub_type);
    };
}