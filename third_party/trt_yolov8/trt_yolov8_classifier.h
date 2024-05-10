
#pragma once

#include <fstream>
#include <iostream>
#include <cmath>
#include <numeric>
#include <opencv2/opencv.hpp>
#include "include/cuda_utils.h"
#include "include/logging.h"
#include "include/model.h"
#include "include/postprocess.h"
#include "include/preprocess.h"
#include "include/utils.h"

namespace trt_yolov8 {
    using namespace nvinfer1;
    class trt_yolov8_classifier
    {
    private:
        void batch_preprocess(std::vector<cv::Mat>& imgs, float* output, int dst_width = 224, int dst_height = 224); 
        std::vector<float> softmax(float *prob, int n);
        std::vector<int> topk(const std::vector<float>& vec, int k);
        void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_input_buffer, float** output_buffer_host);
        void infer(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize);
        void serialize_engine(unsigned int max_batchsize, float& gd, float& gw, std::string& wts_name, std::string& engine_name);
        void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context);

        Logger gLogger;
        const static int kOutputSize = kClsNumClass;
        IRuntime* runtime = nullptr;
        ICudaEngine* engine = nullptr;
        IExecutionContext* context = nullptr;
        cudaStream_t stream;
    public:
        trt_yolov8_classifier(std::string model_path = "");
        ~trt_yolov8_classifier();

        // classify
        void classify(std::vector<cv::Mat> images, std::vector<std::vector<Classification>>& classifications, int top_k = 3);

        // serialize wts to plan file for image classify
        // sub_type: [ n/s/m/l/x/n2/s2/m2/l2/x2/n6/s6/m6/l6/x6 ]
        bool wts_2_engine(std::string wts_name, std::string engine_name, std::string sub_type);
    };
}