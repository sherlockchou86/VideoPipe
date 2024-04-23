#include "../nodes/vp_file_src_node.h"
#include "../nodes/proc/vp_frame_fusion_node.h"
#include "../nodes/vp_split_node.h"
#include "../nodes/vp_placeholder_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## frame_fusion_sample ##
* fuse frames of 2 channels, just merge adjacent frames from 2 channels directly without considering timestamp synchronization.
*/

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_LOGGER_INIT();

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./vp_data/test_video/infrared.mp4");   // source of fusion
    auto file_src_1 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_1", 1, "./vp_data/test_video/rgb.mp4");        // destination of fusion
    // initialize calibration points manually
    std::vector<vp_objects::vp_point> src_cali_points = {vp_objects::vp_point(133, 111), vp_objects::vp_point(338, 110), vp_objects::vp_point(15, 330), vp_objects::vp_point(14, 214)};
    std::vector<vp_objects::vp_point> des_cali_points = {vp_objects::vp_point(1219, 365), vp_objects::vp_point(1787, 367), vp_objects::vp_point(891, 982), vp_objects::vp_point(892, 659)};
    auto fusion = std::make_shared<vp_nodes::vp_frame_fusion_node>("fusion", src_cali_points, des_cali_points);
    auto split = std::make_shared<vp_nodes::vp_split_node>("split", true);
    auto screen_des_0_ori = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0_ori", 0, false);  // original frame for the first channel
    auto screen_des_1_osd = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_1_osd", 1);         // fusion result(osd frame) for the second channel

    // construct pipeline
    fusion->attach_to({file_src_0, file_src_1});
    split->attach_to({fusion});
    screen_des_0_ori->attach_to({split});
    screen_des_1_osd->attach_to({split});

    file_src_0->start();
    file_src_1->start();

    // for debug purpose
    vp_utils::vp_analysis_board board({file_src_0, file_src_1});
    board.display(1, false);

    std::string wait;
    std::getline(std::cin, wait);
    file_src_0->detach_recursively();
    file_src_1->detach_recursively();
}