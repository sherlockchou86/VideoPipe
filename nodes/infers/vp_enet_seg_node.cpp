
#include "vp_enet_seg_node.h"

namespace vp_nodes {

    vp_enet_seg_node::vp_enet_seg_node(std::string node_name, 
                            std::string model_path, 
                            std::string model_config_path, 
                            std::string labels_path, 
                            int input_width, 
                            int input_height, 
                            int batch_size,
                            int class_id_offset,
                            float scale,
                            cv::Scalar mean,
                            cv::Scalar std,
                            bool swap_rb):
                            vp_primary_infer_node(node_name, model_path, model_config_path, labels_path, input_width, input_height, batch_size, class_id_offset, scale, mean, std, swap_rb) {
        this->initialized();
    }

    vp_enet_seg_node::~vp_enet_seg_node() {

    }

    void vp_enet_seg_node::postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        auto& frame_meta = frame_meta_with_batch[0];
        auto& mask = raw_outputs[0];

        // save output as mask directly， this is the mask for the whole frame.
        frame_meta->mask = mask.clone();
    }
}