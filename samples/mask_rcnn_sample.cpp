#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_mask_rcnn_detector_node.h"
#include "../nodes/track/vp_sort_track_node.h"
#include "../nodes/osd/vp_osd_node_v3.h"
#include "../nodes/vp_screen_des_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## mask rcnn sample ##
* image segmentation using mask rcnn.
*/

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./vp_data/test_video/mask_rcnn.mp4", 0.6);
    auto mask_rcnn_detector = std::make_shared<vp_nodes::vp_mask_rcnn_detector_node>("mask_rcnn_detector", "./vp_data/models/mask_rcnn/frozen_inference_graph.pb", "./vp_data/models/mask_rcnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt", "./vp_data/models/coco_80classes.txt");
    auto track_0 = std::make_shared<vp_nodes::vp_sort_track_node>("sort_track_0");
    auto osd_v3_0 = std::make_shared<vp_nodes::vp_osd_node_v3>("osd_v3_0", "./vp_data/font/NotoSansCJKsc-Medium.otf");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);

    // construct pipeline
    mask_rcnn_detector->attach_to({file_src_0});
    track_0->attach_to({mask_rcnn_detector});
    osd_v3_0->attach_to({track_0});
    screen_des_0->attach_to({osd_v3_0});

    file_src_0->start();

    // for debug purpose
    vp_utils::vp_analysis_board board({file_src_0});
    board.display();
}