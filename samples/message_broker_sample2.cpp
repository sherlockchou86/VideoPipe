#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_trt_vehicle_detector.h"
#include "../nodes/infers/vp_trt_vehicle_plate_detector.h"
#include "../nodes/osd/vp_osd_node_v2.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/broker/vp_xml_socket_broker_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## message broker sample2 ##
* show how message broker node works.
* serialize vp_frame_target (vp_sub_target) objects to xml and broke to socket via udp.
*/

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./vp_data/test_video/plate.mp4");
    auto trt_vehicle_detector = std::make_shared<vp_nodes::vp_trt_vehicle_detector>("vehicle_detector", "./vp_data/models/trt/vehicle/vehicle_v8.5.trt");
    auto trt_vehicle_plate_detector = std::make_shared<vp_nodes::vp_trt_vehicle_plate_detector>("vehicle_plate_detector", "./vp_data/models/trt/plate/det_v8.5.trt", "./vp_data/models/trt/plate/rec_v8.5.trt");
    auto xml_socket_broker_0 = std::make_shared<vp_nodes::vp_xml_socket_broker_node>("xml_socket_broker_0", "192.168.77.68", 6666);
    auto osd_0 = std::make_shared<vp_nodes::vp_osd_node_v2>("osd_0", "./vp_data/font/NotoSansCJKsc-Medium.otf");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0, true, vp_objects::vp_size{640, 360});

    // construct pipeline
    trt_vehicle_detector->attach_to({file_src_0});
    trt_vehicle_plate_detector->attach_to({trt_vehicle_detector});
    xml_socket_broker_0->attach_to({trt_vehicle_plate_detector});
    osd_0->attach_to({xml_socket_broker_0});
    screen_des_0->attach_to({osd_0});

    // start pipeline
    file_src_0->start();

    // visualize pipeline for debug
    vp_utils::vp_analysis_board board({file_src_0});
    board.display();
}