
#pragma once

#include "../vp_secondary_infer_node.h"

namespace vp_nodes {
    // common feature encoder for image feature extraction.
    // used for feature extraction, update embeddings of vp_frame_target.
    class vp_feature_encoder_node: public vp_secondary_infer_node
    {
    private:
        /* data */
    public:
        vp_feature_encoder_node(std::string node_name, std::string model_path);
        ~vp_feature_encoder_node();
    };

}