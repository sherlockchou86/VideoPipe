#include "../nodes/vp_file_src_node.h"
#include "../nodes/vp_image_src_node.h"
#include "../nodes/infers/vp_yunet_face_detector_node.h"
#include "../nodes/infers/vp_restoration_node.h"
#include "../nodes/osd/vp_face_osd_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/vp_file_des_node.h"
#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## video_restoration_sample ##
* enhance for any video/images, no training need before running.
*/

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_LOGGER_INIT();

    // create nodes
    auto image_src_0 = std::make_shared<vp_nodes::vp_image_src_node>("image_file_src_0", 0, "./vp_data/test_images/restoration/1/%d.jpg", 3); 
    auto restoration_node = std::make_shared<vp_nodes::vp_restoration_node>("restoration_node", "./vp_data/models/restoration/realesrgan-x4.onnx");
    auto screen_des_0_ori = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0_ori", 0, false);
    auto screen_des_0_osd = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0_osd", 0);

    // construct pipeline
    restoration_node->attach_to({image_src_0});
    screen_des_0_ori->attach_to({restoration_node});
    screen_des_0_osd->attach_to({restoration_node});

    // start pipeline
    image_src_0->start();

    // for debug purpose
    vp_utils::vp_analysis_board board({image_src_0});
    board.display(1, false);

    std::string wait;
    std::getline(std::cin, wait);
    image_src_0->detach_recursively();
}