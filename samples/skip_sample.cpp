#include "../nodes/vp_file_src_node.h"
#include "../nodes/vp_rtsp_src_node.h"
#include "../nodes/infers/vp_yolo_detector_node.h"
#include "../nodes/osd/vp_osd_node.h"
#include "../nodes/vp_split_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/vp_rtmp_des_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## skip_sample ##
* 2 inputs , and skip 2 frames every 3 frames for the 2nd channel.
*/

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./vp_data/test_video/vehicle_count.mp4", 0.5);
    auto rtsp_src_1 = std::make_shared<vp_nodes::vp_rtsp_src_node>("rtsp_src_1", 1, "rtsp://admin:admin12345@192.168.3.157", 0.4, "avdec_h264", 2);  // skip 2 frames every 3 frames
    auto yolo_detector = std::make_shared<vp_nodes::vp_yolo_detector_node>("yolo_detector", "./vp_data/models/det_cls/yolov3-tiny-2022-0721_best.weights", "./vp_data/models/det_cls/yolov3-tiny-2022-0721.cfg", "./vp_data/models/det_cls/yolov3_tiny_5classes.txt");
    auto split = std::make_shared<vp_nodes::vp_split_node>("split", true);
    auto osd = std::make_shared<vp_nodes::vp_osd_node>("osd", "./vp_data/font/NotoSansCJKsc-Medium.otf");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);
    auto rtmp_des_1 = std::make_shared<vp_nodes::vp_rtmp_des_node>("rtmp_des_0", 1, "rtmp://192.168.77.60/live/10000");

    // construct pipeline
    yolo_detector->attach_to({file_src_0, rtsp_src_1});
    osd->attach_to({yolo_detector});
    split->attach_to({osd});
    screen_des_0->attach_to({split});
    rtmp_des_1->attach_to({split});

    // start channels
    file_src_0->start();
    rtsp_src_1->start();

    // for debug purpose
    vp_utils::vp_analysis_board board({file_src_0, rtsp_src_1});
    board.display();
}