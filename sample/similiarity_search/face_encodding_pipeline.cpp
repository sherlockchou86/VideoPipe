#include "../VP.h"

#include "../../nodes/vp_image_src_node.h"
#include "../../nodes/infers/vp_yunet_face_detector_node.h"
#include "../../nodes/infers/vp_sface_feature_encoder_node.h"
#include "../../nodes/broker/vp_embeddings_socket_broker_node.h"
#include "../../nodes/osd/vp_face_osd_node.h"
#include "../../nodes/vp_screen_des_node.h"

#include "../../utils/analysis_board/vp_analysis_board.h"

/*
* ## face_similiarity_search_sample ##
* generate embedding data for faces using VideoPipe, then send them via socket.
*/

#if face_similiarity_search_sample

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto image_src_0 = std::make_shared<vp_nodes::vp_image_src_node>("image_src_0", 0, "../images/faces/%d.jpg");
    auto yunet_face_detector_0 = std::make_shared<vp_nodes::vp_yunet_face_detector_node>("yunet_face_detector_0", "../models/face/face_detection_yunet_2022mar.onnx");
    auto sface_face_encoder_0 = std::make_shared<vp_nodes::vp_sface_feature_encoder_node>("sface_face_encoder_0", "../models/face/face_recognition_sface_2021dec.onnx");
    auto embedding_broker = std::make_shared<vp_nodes::vp_embeddings_socket_broker_node>("embedding_broker", "192.168.77.68", 8888, "static/cropped_images", 40, 40, vp_nodes::vp_broke_for::FACE);
    auto osd_0 = std::make_shared<vp_nodes::vp_face_osd_node>("osd_0");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);
    
    // construct pipeline
    yunet_face_detector_0->attach_to({image_src_0});
    sface_face_encoder_0->attach_to({yunet_face_detector_0});
    embedding_broker->attach_to({sface_face_encoder_0});
    osd_0->attach_to({embedding_broker});
    screen_des_0->attach_to({osd_0});

    // start pipeline
    image_src_0->start();

    // visualize pipeline for debug
    vp_utils::vp_analysis_board board({image_src_0});
    board.display();
}

#endif