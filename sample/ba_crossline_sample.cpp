#include "VP.h"

#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_yolo_detector_node.h"
#include "../nodes/track/vp_sort_track_node.h"
#include "../nodes/ba/vp_ba_crossline_node.h"
#include "../nodes/osd/vp_ba_crossline_osd_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/vp_rtmp_des_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## ba crossline sample ##
* behaviour analysis for crossline.
*/

#if ba_crossline_sample

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./test_video/vehicle_count.mp4", 0.4);
    auto yolo_detector = std::make_shared<vp_nodes::vp_yolo_detector_node>("yolo_detector", "models/det_cls/yolov3-tiny-2022-0721_best.weights", "models/det_cls/yolov3-tiny-2022-0721.cfg", "models/det_cls/yolov3_tiny_5classes.txt");
    auto tracker = std::make_shared<vp_nodes::vp_sort_track_node>("sort_tracker");
    
    // define a line in frame for every channel (value MUST in the scope of frame'size)
    vp_objects::vp_point start(0, 250);  // change to proper value
    vp_objects::vp_point end(700, 220);  // change to proper value
    std::map<int, vp_objects::vp_line> lines = {{0, vp_objects::vp_line(start, end)}};  // channel0 -> line
    auto ba_crossline = std::make_shared<vp_nodes::vp_ba_crossline_node>("ba_crossline", lines);
    auto osd = std::make_shared<vp_nodes::vp_ba_crossline_osd_node>("osd");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);
    auto rtmp_des_0 = std::make_shared<vp_nodes::vp_rtmp_des_node>("rtmp_des_0", 0, "rtmp://192.168.77.60/live/9000");
    
    // construct pipeline
    yolo_detector->attach_to({file_src_0});
    tracker->attach_to({yolo_detector});
    ba_crossline->attach_to({tracker});
    osd->attach_to({ba_crossline});
    screen_des_0->attach_to({osd});
    rtmp_des_0->attach_to({osd});

    file_src_0->start();

    // for debug purpose
    vp_utils::vp_analysis_board board({file_src_0});
    board.display(1, false);

    std::string wait;
    std::getline(std::cin, wait);
    file_src_0->detach_recursively();
}

#endif