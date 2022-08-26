
#include "vp_primary_infer_node.h"

namespace vp_nodes {
        
    vp_primary_infer_node::vp_primary_infer_node(std::string node_name, 
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
                                                vp_infer_node(node_name, 
                                                            vp_infer_type::PRIMARY, 
                                                            model_path, 
                                                            model_config_path, 
                                                            labels_path,
                                                            input_width, 
                                                            input_height, 
                                                            batch_size,
                                                            scale,
                                                            mean,
                                                            std,
                                                            swap_rb),
                                                class_id_offset(class_id_offset) {
    }
    
    vp_primary_infer_node::~vp_primary_infer_node() {

    }

    void vp_primary_infer_node::prepare(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch, std::vector<cv::Mat>& mats_to_infer) {
        // fetch the whole frame, can batch by batch
        for (auto& i: frame_meta_with_batch) {
            mats_to_infer.push_back(i->frame);
        }
    }
}