#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_yunet_face_detector_node.h"
#include "../nodes/infers/vp_face_swap_node.h"
#include "../nodes/osd/vp_face_osd_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/vp_file_des_node.h"
#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## face_swap_sample ##
* swap face for any video/images, no training need before running.
*/

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_LOGGER_INIT();

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./vp_data/test_video/face.mp4", 1.0, true, "avdec_h264", 4);
    auto yunet_face_detector = std::make_shared<vp_nodes::vp_yunet_face_detector_node>("yunet_face_detector", "./vp_data/models/face/face_detection_yunet_2022mar.onnx");
    auto face_swap = std::make_shared<vp_nodes::vp_face_swap_node>("face_swap", "./vp_data/models/face/face_detection_yunet_2022mar.onnx", "./vp_data/models/face/swap/w600k_r50.onnx", "./vp_data/models/face/swap/emap.txt", "./vp_data/models/face/swap/inswapper_128.onnx", "./github/inswapper/data/mans1.jpeg");
    //auto osd = std::make_shared<vp_nodes::vp_face_osd_node>("osd");
    auto screen_des_0_ori = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0_ori", 0, false);
    auto screen_des_0_osd = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0_osd", 0);
    auto file_des_node_0 = std::make_shared<vp_nodes::vp_file_des_node>("file_des_0", 0, ".", "", 1);

    // construct pipeline
    yunet_face_detector->attach_to({file_src_0});
    face_swap->attach_to({yunet_face_detector});
    //osd->attach_to({face_swap});
    screen_des_0_ori->attach_to({face_swap});
    screen_des_0_osd->attach_to({face_swap});
    file_des_node_0->attach_to({face_swap}); // save swap result to file

    file_src_0->start();

    // for debug purpose
    vp_utils::vp_analysis_board board({file_src_0});
    board.display(1, false);

    std::string wait;
    std::getline(std::cin, wait);
    file_src_0->detach_recursively();
}