
#pragma once

#include "../vp_primary_infer_node.h"
#include "../../objects/vp_frame_target.h"



namespace vp_nodes {
    // driving area segmentation based on yolov5s-seg-v7.0
    // https://github.com/ultralytics/yolov5/releases/tag/v7.0
    class vp_yolo5_seg_node: public vp_primary_infer_node
    {
    private:
        /* data */
        float conf_threshold = 0.5;
        float iou_threshold = 0.7;
        int MASK_CHANNELS = 32;
        cv::Mat scale_ratio;
        cv::Point padding;
        cv::Mat letterbox(const cv::Mat& src, cv::Size target_size, cv::Mat& scale_ratio, cv::Point& padding);
        cv::Mat sigmoid(const cv::Mat& x);
        cv::Mat process_mask(const cv::Mat& proto,  // [1, 32, H/4, W/4]
                    const cv::Mat& coeffs,          // [1, 32]
                    const cv::Rect& bbox,           
                    const cv::Size& input_size,     // e.g., 640x384
                    const cv::Point& padding,
                    const float x_scale,
                    const float y_scale);
        std::vector<int> nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, float iou_threshold);
    protected:
        virtual void infer(const cv::Mat& blob_to_infer, std::vector<cv::Mat>& raw_outputs) override;
        virtual void preprocess(const std::vector<cv::Mat>& mats_to_infer, cv::Mat& blob_to_infer) override;
        virtual void postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
        virtual void prepare(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch, std::vector<cv::Mat>& mats_to_infer) override;
    public:
        vp_yolo5_seg_node(std::string node_name, 
                            std::string model_path, 
                            std::string labels_path = "", 
                            int input_width = 640, 
                            int input_height = 384, 
                            int batch_size = 1,
                            int class_id_offset = 0,
                            float conf_threshold = 0.5,
                            float iou_threshold = 0.7);
        ~vp_yolo5_seg_node();
    };    
}