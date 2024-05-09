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
    class trt_yolov8_seg_detector
    {
    private:
        /* data */
        cv::Rect get_downscale_rect(float bbox[4], float scale);
        std::vector<cv::Mat> process_mask(const float* proto, int proto_size, std::vector<Detection>& dets);
        void serialize_engine(std::string& wts_name, std::string& engine_name, std::string& sub_type, float& gd, float& gw,
                            int& max_channels);
        void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine,
                                IExecutionContext** context);
        void prepare_buffer(ICudaEngine* engine, float** input_buffer_device, float** output_buffer_device,
                            float** output_seg_buffer_device, float** output_buffer_host, float** output_seg_buffer_host,
                            float** decode_ptr_host, float** decode_ptr_device, std::string cuda_post_process);
        void infer(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output, float* output_seg,
                int batchsize, float* decode_ptr_host, float* decode_ptr_device, int model_bboxes,
                std::string cuda_post_process);

        Logger gLogger;
        const int kOutputSize = kMaxNumOutputBbox * (sizeof(Detection) - sizeof(float) * 51) / sizeof(float) + 1;
        const static int kOutputSegSize = 32 * (kInputH / 4) * (kInputW / 4);
    
        std::string cuda_post_process = "c";
        int model_bboxes;
        cudaStream_t stream;

        // Deserialize the engine from file
        nvinfer1::IRuntime* runtime = nullptr;
        nvinfer1::ICudaEngine* engine = nullptr;
        nvinfer1::IExecutionContext* context = nullptr;
    public:
        trt_yolov8_seg_detector(std::string model_path = "");
        ~trt_yolov8_seg_detector();

        // detect 
        void detect(std::vector<cv::Mat> images, std::vector<std::vector<Detection>>& detections, std::vector<std::vector<cv::Mat>>& masks);

        // serialize wts to plan file for segment
        // sub_type: [ n/s/m/l/x/n2/s2/m2/l2/x2/n6/s6/m6/l6/x6 ]
        bool wts_2_engine(std::string wts_name, std::string engine_name, std::string sub_type);
    };
}