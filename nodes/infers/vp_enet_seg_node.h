#pragma once

#include "../vp_primary_infer_node.h"


namespace vp_nodes {
    // semantic segmentation based on ENet
    // 
    class vp_enet_seg_node: public vp_primary_infer_node
    {
    private:
        /* data */
    protected:
        virtual void postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
    public:
        vp_enet_seg_node(std::string node_name, 
                            std::string model_path, 
                            std::string model_config_path = "", 
                            std::string labels_path = "", 
                            int input_width = 1024, 
                            int input_height = 512, 
                            int batch_size = 1,
                            int class_id_offset = 0,
                            float scale = 1 / 255.0,
                            cv::Scalar mean = cv::Scalar(0),
                            cv::Scalar std = cv::Scalar(1),
                            bool swap_rb = true);
        ~vp_enet_seg_node();
    };
}