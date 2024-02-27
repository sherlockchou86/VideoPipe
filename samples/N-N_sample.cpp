#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_yunet_face_detector_node.h"
#include "../nodes/infers/vp_sface_feature_encoder_node.h"
#include "../nodes/osd/vp_face_osd_node_v2.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/vp_rtmp_des_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## N-N sample ##
* multi pipe exist separately and each pipe is 1-1-1 (can be any structure like 1-1-N, 1-N-N)
*/

int main() {
    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_LOGGER_INIT();

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./vp_data/test_video/face.mp4", 0.6);
    auto yunet_face_detector_0 = std::make_shared<vp_nodes::vp_yunet_face_detector_node>("yunet_face_detector_0", "./vp_data/models/face/face_detection_yunet_2022mar.onnx");
    auto sface_face_encoder_0 = std::make_shared<vp_nodes::vp_sface_feature_encoder_node>("sface_face_encoder_0", "./vp_data/models/face/face_recognition_sface_2021dec.onnx");
    auto osd_0 = std::make_shared<vp_nodes::vp_face_osd_node_v2>("osd_0");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);

    auto file_src_1 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_1", 0, "./vp_data/test_video/face2.mp4");
    auto yunet_face_detector_1 = std::make_shared<vp_nodes::vp_yunet_face_detector_node>("yunet_face_detector_1", "./vp_data/models/face/face_detection_yunet_2022mar.onnx");
    auto sface_face_encoder_1 = std::make_shared<vp_nodes::vp_sface_feature_encoder_node>("sface_face_encoder_1", "./vp_data/models/face/face_recognition_sface_2021dec.onnx");
    auto osd_1 = std::make_shared<vp_nodes::vp_face_osd_node_v2>("osd_1");
    auto screen_des_1 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_1", 0);

    // construct pipeline
    // pipe 0
    yunet_face_detector_0->attach_to({file_src_0});
    sface_face_encoder_0->attach_to({yunet_face_detector_0});
    osd_0->attach_to({sface_face_encoder_0});
    screen_des_0->attach_to({osd_0});

    // pipe 1
    yunet_face_detector_1->attach_to({file_src_1});
    sface_face_encoder_1->attach_to({yunet_face_detector_1});
    osd_1->attach_to({sface_face_encoder_1});
    screen_des_1->attach_to({osd_1});

    file_src_0->start();
    file_src_1->start();

    // for debug purpose
    vp_utils::vp_analysis_board board({file_src_0, file_src_1});
    board.display();
}