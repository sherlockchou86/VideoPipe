#include "VP.h"

#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_trt_vehicle_detector.h"
#include "../nodes/track/vp_sort_track_node.h"
#include "../nodes/ba/vp_ba_stop_node.h"
#include "../nodes/osd/vp_ba_stop_osd_node.h"
#include "../nodes/vp_split_node.h"
#include "../nodes/vp_screen_des_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## ba stop sample ##
* behaviour analysis for stop, single instance of ba node work on 2 channels.
*/

#if ba_stop_sample

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./test_video/vehicle_stop2.mp4", 0.6);
    auto file_src_1 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_1", 1, "./test_video/vehicle_stop3.mp4", 0.6);
    auto trt_vehicle_detector = std::make_shared<vp_nodes::vp_trt_vehicle_detector>("vehicle_detector", "./models/trt/vehicle/vehicle_v8.5.trt");
    auto tracker = std::make_shared<vp_nodes::vp_sort_track_node>("sort_tracker");
    
    // define a region in frame for every channel (value MUST in the scope of frame'size)
    std::map<int, std::vector<vp_objects::vp_point>> regions = {
        {0, std::vector<vp_objects::vp_point>{vp_objects::vp_point(20, 30), vp_objects::vp_point(600, 40), vp_objects::vp_point(600, 300), vp_objects::vp_point(10, 300)}},  // channel0 -> region
        {1, std::vector<vp_objects::vp_point>{vp_objects::vp_point(20, 30), vp_objects::vp_point(1000, 40), vp_objects::vp_point(1000, 600), vp_objects::vp_point(10, 600)}}   // channel1 -> region
    };
    auto ba_stop = std::make_shared<vp_nodes::vp_ba_stop_node>("ba_stop", regions);
    auto osd = std::make_shared<vp_nodes::vp_ba_stop_osd_node>("osd");
    auto split = std::make_shared<vp_nodes::vp_split_node>("split", true);
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);
    auto screen_des_1 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_1", 1);
    
    // construct pipeline
    trt_vehicle_detector->attach_to({file_src_0, file_src_1});
    tracker->attach_to({trt_vehicle_detector});
    ba_stop->attach_to({tracker});
    osd->attach_to({ba_stop});
    split->attach_to({osd});
    screen_des_0->attach_to({split});
    screen_des_1->attach_to({split});

    file_src_0->start();
    file_src_1->start();

    // for debug purpose
    vp_utils::vp_analysis_board board({file_src_0, file_src_1});
    board.display();
}

#endif