#pragma once

#include "../vp_primary_infer_node.h"


namespace vp_nodes {
    // lane detect based on CenterNet
    // 
    class vp_lane_detector_node: public vp_primary_infer_node
    {
    private:
        /* data */
    protected:
        virtual void postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
    public:
        vp_lane_detector_node(std::string node_name, 
                            std::string model_path, 
                            std::string model_config_path = "", 
                            std::string labels_path = "", 
                            int input_width = 736, 
                            int input_height = 416, 
                            int batch_size = 1,
                            int class_id_offset = 0,
                            float scale = 1,
                            cv::Scalar mean = cv::Scalar(0),
                            cv::Scalar std = cv::Scalar(1),
                            bool swap_rb = true,
                            bool swap_chn = true);
        ~vp_lane_detector_node();
    };
}