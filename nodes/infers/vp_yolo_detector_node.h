
#pragma once

#include "../vp_primary_infer_node.h"

namespace vp_nodes {
    // yolo detector, support yolov3/4/5
    // https://github.com/pjreddie/darknet
    class vp_yolo_detector_node: public vp_primary_infer_node
    {
    private:
        float score_threshold;
        float confidence_threshold;
        float nms_threshold;
    protected:
        virtual void postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
    public:
        vp_yolo_detector_node(std::string node_name, 
                            std::string model_path, 
                            std::string model_config_path = "", 
                            std::string labels_path = "", 
                            int input_width = 416, 
                            int input_height = 416, 
                            int batch_size = 1,
                            int class_id_offset = 0,
                            float score_threshold = 0.5,
                            float confidence_threshold = 0.5,
                            float nms_threshold = 0.5,
                            float scale = 1 / 255.0,
                            cv::Scalar mean = cv::Scalar(0),
                            cv::Scalar std = cv::Scalar(1),
                            bool swap_rb = true);
        ~vp_yolo_detector_node();
    };
}