
#pragma once

#include "../vp_secondary_infer_node.h"

namespace vp_nodes {
    // face feature encoder based on SFace, update embeddings of vp_frame_face_target
    // https://github.com/opencv/opencv/blob/4.x/modules/objdetect/src/face_recognize.cpp
    // https://github.com/zhongyy/SFace
    class vp_sface_feature_encoder_node: public vp_secondary_infer_node
    {
    private:
        /* data */
    public:
        vp_sface_feature_encoder_node(std::string node_name, std::string model_path);
        ~vp_sface_feature_encoder_node();
    };

 }