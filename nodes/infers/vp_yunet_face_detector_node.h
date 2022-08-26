
#pragma once

#include "../vp_primary_infer_node.h"
#include "../../objects/vp_frame_face_target.h"

namespace vp_nodes {
    // face detector based on YunNet
    // https://github.com/opencv/opencv/blob/4.x/modules/objdetect/src/face_detect.cpp
    // https://github.com/ShiqiYu/libfacedetection
    class vp_yunet_face_detector_node: public vp_primary_infer_node
    {
    private:
        // names of output layers in yunet
        const std::vector<std::string> out_names = {"loc", "conf", "iou"};
        float scoreThreshold = 0.7;
        float nmsThreshold = 0.5;
        int topK = 50;
        int inputW;
        int inputH;
        std::vector<cv::Rect2f> priors;
        void generatePriors();
    protected:
        // override infer and preprocess as yunet has a different logic
        virtual void infer(const cv::Mat& blob_to_infer, std::vector<cv::Mat>& raw_outputs) override;
        virtual void preprocess(const std::vector<cv::Mat>& mats_to_infer, cv::Mat& blob_to_infer) override;

        virtual void postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
    public:
        vp_yunet_face_detector_node(std::string node_name, std::string model_path, float score_threshold = 0.7, float nms_threshold = 0.5, int top_k = 50);
        ~vp_yunet_face_detector_node();
    };

}