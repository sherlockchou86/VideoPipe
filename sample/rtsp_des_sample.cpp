
#include "VP.h"

#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_yolo_detector_node.h"
#include "../nodes/osd/vp_osd_node.h"
#include "../nodes/vp_split_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/vp_rtsp_des_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## rtsp_des_sample ##
* show how vp_rtsp_des_node works, push video stream via rtsp, no specialized rtsp server needed.
* visit `rtsp://server-ip:rtsp_port/rtsp_name directly.
*/

#if rtsp_des_sample

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "test_video/vehicle_count.mp4", 0.5);
    auto file_src_1 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_1", 1, "test_video/vehicle_stop.mp4", 0.5);
    auto yolo_detector = std::make_shared<vp_nodes::vp_yolo_detector_node>("yolo_detector", "models/det_cls/yolov3-tiny-2022-0721_best.weights", "models/det_cls/yolov3-tiny-2022-0721.cfg", "models/det_cls/yolov3_tiny_5classes.txt");
    auto osd = std::make_shared<vp_nodes::vp_osd_node>("osd");
    auto split = std::make_shared<vp_nodes::vp_split_node>("split_by_channel", true);
    //auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);
    auto rtsp_des_0 = std::make_shared<vp_nodes::vp_rtsp_des_node>("rtsp_des_0", 0, 8000, "rtsp_0");
    
    //auto srceen_des_1 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_1", 1);
    auto rtsp_des_1 = std::make_shared<vp_nodes::vp_rtsp_des_node>("rtsp_des_1", 1, 8000, "rtsp_1");

    // construct pipeline
    yolo_detector->attach_to({file_src_0, file_src_1});
    osd->attach_to({yolo_detector});
    split->attach_to({osd});
    rtsp_des_0->attach_to({split});
    rtsp_des_1->attach_to({split});

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

#endif