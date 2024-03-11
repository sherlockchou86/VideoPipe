#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_yunet_face_detector_node.h"
#include "../nodes/infers/vp_sface_feature_encoder_node.h"
#include "../nodes/osd/vp_face_osd_node_v2.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/vp_rtmp_des_node.h"
#include "../nodes/vp_split_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## 1-N-N sample ##
* 1 video input and then split into 2 branches for different infer tasks, then 2 total outputs(no need to sync in such situations).
*/

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./vp_data/test_video/face.mp4", 0.6);
    auto split = std::make_shared<vp_nodes::vp_split_node>("split", false, true);  // split by deep-copy not by channel!

    // branch a
    auto yunet_face_detector_a = std::make_shared<vp_nodes::vp_yunet_face_detector_node>("yunet_face_detector_a", "./vp_data/models/face/face_detection_yunet_2022mar.onnx");
    auto sface_face_encoder_a = std::make_shared<vp_nodes::vp_sface_feature_encoder_node>("sface_face_encoder_a", "./vp_data/models/face/face_recognition_sface_2021dec.onnx");
    auto osd_a = std::make_shared<vp_nodes::vp_face_osd_node_v2>("osd_a");
    auto screen_des_a = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_a", 0);

    // branch b
    auto yunet_face_detector_b = std::make_shared<vp_nodes::vp_yunet_face_detector_node>("yunet_face_detector_b", "./vp_data/models/face/face_detection_yunet_2022mar.onnx");
    auto sface_face_encoder_b = std::make_shared<vp_nodes::vp_sface_feature_encoder_node>("sface_face_encoder_b", "./vp_data/models/face/face_recognition_sface_2021dec.onnx");
    auto osd_b = std::make_shared<vp_nodes::vp_face_osd_node_v2>("osd_b");
    auto screen_des_b = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_b", 0);

    // construct pipeline
    split->attach_to({file_src_0});

    // branch a
    yunet_face_detector_a->attach_to({split});
    sface_face_encoder_a->attach_to({yunet_face_detector_a});
    osd_a->attach_to({sface_face_encoder_a});
    screen_des_a->attach_to({osd_a});

    // branch b
    yunet_face_detector_b->attach_to({split});
    sface_face_encoder_b->attach_to({yunet_face_detector_b});
    osd_b->attach_to({sface_face_encoder_b});
    screen_des_b->attach_to({osd_b});

    file_src_0->start();

    // for debug purpose
    vp_utils::vp_analysis_board board({file_src_0});
    board.display(1, false);

    std::string wait;
    std::getline(std::cin, wait);
    file_src_0->detach_recursively();
}