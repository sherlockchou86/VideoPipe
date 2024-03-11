#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_trt_vehicle_detector.h"
#include "../nodes/infers/vp_trt_vehicle_color_classifier.h"
#include "../nodes/infers/vp_trt_vehicle_type_classifier.h"
#include "../nodes/infers/vp_trt_vehicle_feature_encoder.h"
#include "../nodes/track/vp_sort_track_node.h"
#include "../nodes/vp_sync_node.h"
#include "../nodes/vp_split_node.h"
#include "../nodes/osd/vp_osd_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## 1-N-1_sample_sample ##
* 1 input video, detect vehicles first，classify by colors and tracking vehicles in parallel, then sync data and output on screen.
*/

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::DEBUG);
    VP_SET_LOG_KEYWORDS_FOR_DEBUG({"sync"});  // only write debug log for sync node
    VP_LOGGER_INIT();

    // create nodes for 1-N-1 pipeline
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./vp_data/test_video/jam2.mp4");
    auto trt_vehicle_detector_0 = std::make_shared<vp_nodes::vp_trt_vehicle_detector>("trt_detector_0", "./vp_data/models/trt/vehicle/vehicle_v8.5.trt");
    auto split = std::make_shared<vp_nodes::vp_split_node>("split", false, true);  // split by deep-copy not by channel!
    auto trt_vehicle_color_classifier_0 = std::make_shared<vp_nodes::vp_trt_vehicle_color_classifier>("trt_color_cls_0", "./vp_data/models/trt/vehicle/vehicle_color_v8.5.trt", std::vector<int>{0, 1, 2});
    auto track_0 = std::make_shared<vp_nodes::vp_sort_track_node>("track_0");
    auto sync = std::make_shared<vp_nodes::vp_sync_node>("sync", vp_nodes::vp_sync_mode::UPDATE, 160);
    auto osd_0 = std::make_shared<vp_nodes::vp_osd_node>("osd_0");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);
    
    // construct 1-N-1 pipeline
    trt_vehicle_detector_0->attach_to({file_src_0});
    split->attach_to({trt_vehicle_detector_0});
    trt_vehicle_color_classifier_0->attach_to({split});
    track_0->attach_to({split});
    sync->attach_to({trt_vehicle_color_classifier_0, track_0});
    osd_0->attach_to({sync});
    screen_des_0->attach_to({osd_0});
    /*
      the format of pipeline:
                                                     / trt_vehicle_color_classifier_0  \ 
      file_src_0 -> trt_vehicle_detector_0 -> split ->                                 -> sync -> osd_0 -> screen_des_0
                                                     \ track_0                         /
    */

    // start 1-N-1 pipeline
    file_src_0->start();

    // visualize pipelines for debug
    vp_utils::vp_analysis_board board({file_src_0});
    board.display();
}