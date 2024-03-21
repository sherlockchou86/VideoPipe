#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_yolo_detector_node.h"
#include "../nodes/osd/vp_osd_node.h"
#include "../nodes/vp_split_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/vp_rtsp_des_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## multi_detectors_sample ##
* detect obstacles AND vehicles using yolo on road.
*/

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./vp_data/test_video/unclear.mp4", 0.5);
    auto file_src_1 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_1", 1, "./vp_data/test_video/roadblock.mp4", 0.6);
    auto obstacle_detector = std::make_shared<vp_nodes::vp_yolo_detector_node>("obstacle_detector", "./vp_data/models/det_cls/obstacles_yolov5s.onnx", "", "./vp_data/models/det_cls/obstacles_2classes.txt", 640, 640);
    // MUST set class_id_offset for the 2nd detector which is equal with total classes of the 1st detectors
    auto vehicle_detector = std::make_shared<vp_nodes::vp_yolo_detector_node>("vehicle_detector", "./vp_data/models/det_cls/yolov3-tiny-2022-0721_best.weights", "./vp_data/models/det_cls/yolov3-tiny-2022-0721.cfg", "./vp_data/models/det_cls/yolov3_tiny_5classes.txt", 416, 416, 1, 2);
    auto osd = std::make_shared<vp_nodes::vp_osd_node>("osd");
    auto split = std::make_shared<vp_nodes::vp_split_node>("split_by_channel", true);
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);    
    auto srceen_des_1 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_1", 1);

    // construct pipeline
    obstacle_detector->attach_to({file_src_0, file_src_1});
    vehicle_detector->attach_to({obstacle_detector});
    osd->attach_to({vehicle_detector});
    split->attach_to({osd});
    screen_des_0->attach_to({split});
    srceen_des_1->attach_to({split});

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