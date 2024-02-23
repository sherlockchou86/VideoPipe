
#include "vp_feature_encoder_node.h"


namespace vp_nodes {
    
    vp_feature_encoder_node::vp_feature_encoder_node(std::string node_name, std::string model_path):
                                                    vp_secondary_infer_node(node_name, model_path) {
        this->initialized();
    }
    
    vp_feature_encoder_node::~vp_feature_encoder_node() {
        deinitialized();        
    } 
}