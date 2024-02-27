#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_openpose_detector_node.h"
#include "../nodes/osd/vp_pose_osd_node.h"
#include "../nodes/vp_screen_des_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## openpose sample ##
* pose estimation by OpenPose network.
*/

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./vp_data/test_video/pose.mp4");
    auto openpose_detector = std::make_shared<vp_nodes::vp_openpose_detector_node>("openpose_detector", "./vp_data/models/openpose/pose/body_25_pose_iter_584000.caffemodel", "./vp_data/models/openpose/pose/body_25_pose_deploy.prototxt", "", 368, 368, 1, 0, 0.1, vp_objects::vp_pose_type::body_25);
    auto pose_osd_0 = std::make_shared<vp_nodes::vp_pose_osd_node>("pose_osd_0");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);

    // construct pipeline
    openpose_detector->attach_to({file_src_0});
    pose_osd_0->attach_to({openpose_detector});
    screen_des_0->attach_to({pose_osd_0});

    file_src_0->start();

    // for debug purpose
    vp_utils::vp_analysis_board board({file_src_0});
    board.display();
}