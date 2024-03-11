#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_trt_vehicle_detector.h"
#include "../nodes/infers/vp_trt_vehicle_color_classifier.h"
#include "../nodes/infers/vp_trt_vehicle_plate_detector_v2.h"
#include "../nodes/track/vp_sort_track_node.h"
#include "../nodes/vp_sync_node.h"
#include "../nodes/vp_split_node.h"
#include "../nodes/osd/vp_osd_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## 1-N-1_sample_sample ##
* 1 input video, detect vehicles + classify by colors, detect license plate + track by sort, 2 tasks in parallel，then sync data and output on screen.
*/

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::DEBUG);
    VP_SET_LOG_KEYWORDS_FOR_DEBUG({"sync"});  // only write debug log for sync node
    VP_LOGGER_INIT();

    // create nodes for 1-N-1 pipeline
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./vp_data/test_video/plate2.mp4");
    auto split = std::make_shared<vp_nodes::vp_split_node>("split", false, true);  // split by deep-copy not by channel!
    auto vehicle_detector = std::make_shared<vp_nodes::vp_trt_vehicle_detector>("vehicle_detector", "./vp_data/models/trt/vehicle/vehicle_v8.5.trt");
    auto vehicle_color_cls = std::make_shared<vp_nodes::vp_trt_vehicle_color_classifier>("vehicle_color_cls", "./vp_data/models/trt/vehicle/vehicle_color_v8.5.trt", std::vector<int>{0, 1, 2});
    auto plate_detector = std::make_shared<vp_nodes::vp_trt_vehicle_plate_detector_v2>("plate_detector", "./vp_data/models/trt/plate/det_v8.5.trt", "./vp_data/models/trt/plate/rec_v8.5.trt");
    auto plate_tracker = std::make_shared<vp_nodes::vp_sort_track_node>("plate_tracker");
    auto sync = std::make_shared<vp_nodes::vp_sync_node>("sync");
    auto osd = std::make_shared<vp_nodes::vp_osd_node>("osd", "./vp_data/font/NotoSansCJKsc-Medium.otf");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);
    
    // construct 1-N-1 pipeline
    split->attach_to({file_src_0});
    vehicle_detector->attach_to({split});
    plate_detector->attach_to({split});
    vehicle_color_cls->attach_to({vehicle_detector});
    plate_tracker->attach_to({plate_detector});
    sync->attach_to({vehicle_color_cls, plate_tracker});
    osd->attach_to({sync});
    screen_des_0->attach_to({osd});
    /*
      the format of pipeline:
                            / vehicle_detector -> vehicle_color_cls  \ 
      file_src_0 -> split ->                                         -> sync -> osd -> screen_des_0
                            \ plate_detector -> plate_tracker        /
    */

    // start 1-N-1 pipeline
    file_src_0->start();

    // visualize pipelines for debug
    vp_utils::vp_analysis_board board({file_src_0});
    board.display();
}