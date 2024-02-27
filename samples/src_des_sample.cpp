#include "../nodes/vp_file_src_node.h"
#include "../nodes/vp_rtsp_src_node.h"
#include "../nodes/vp_udp_src_node.h"

#include "../nodes/vp_screen_des_node.h"
#include "../nodes/vp_rtmp_des_node.h"
#include "../nodes/vp_fake_des_node.h"
#include "../nodes/vp_file_des_node.h"

#include "../nodes/infers/vp_trt_vehicle_detector.h"
#include "../nodes/osd/vp_osd_node.h"
#include "../nodes/vp_split_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## src des sample ##
* show how src nodes and des nodes work
* 3 (file, rtsp, udp) input and merge into 1 infer task, then resume to 3 branches for outputs (screen, rtmp, fake)
*/

int main() {

    // log config
    // ...
    VP_LOGGER_INIT();

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./vp_data/test_video/vehicle_count.mp4", 0.5);
    auto rtsp_src_1 = std::make_shared<vp_nodes::vp_rtsp_src_node>("rtsp_src_1", 1, "rtsp://admin:admin12345@192.168.77.203", 0.4);
    auto udp_src_2 = std::make_shared<vp_nodes::vp_udp_src_node>("udp_src_2", 2, 6000, 0.3);

    auto trt_vehicle_detector = std::make_shared<vp_nodes::vp_trt_vehicle_detector>("trt_vehicle_detector","./vp_data/models/trt/vehicle/vehicle_v8.5.trt");

    auto split = std::make_shared<vp_nodes::vp_split_node>("", true);

    auto osd_0 = std::make_shared<vp_nodes::vp_osd_node>("osd_0", "./vp_data/font/NotoSansCJKsc-Medium.otf");
    auto osd_1 = std::make_shared<vp_nodes::vp_osd_node>("osd_1", "./vp_data/font/NotoSansCJKsc-Medium.otf");
    auto osd_2 = std::make_shared<vp_nodes::vp_osd_node>("osd_2", "./vp_data/font/NotoSansCJKsc-Medium.otf");

    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);
    auto rtmp_des_1 = std::make_shared<vp_nodes::vp_rtmp_des_node>("rtmp_des_0", 1, "rtmp://192.168.77.60/live/10000");
    auto fake_des_2 = std::make_shared<vp_nodes::vp_fake_des_node>("fake_des_2", 2);

    // construct pipeline
    // auto merge
    trt_vehicle_detector->attach_to({file_src_0, rtsp_src_1, udp_src_2});

    split->attach_to({trt_vehicle_detector});

    // resume to 3 branches
    osd_0->attach_to({split});
    osd_1->attach_to({split});
    osd_2->attach_to({split});

    screen_des_0->attach_to({osd_0});
    rtmp_des_1->attach_to({osd_1});
    fake_des_2->attach_to({osd_2});

    // start channels
    file_src_0->start();
    rtsp_src_1->start();
    udp_src_2->start();

    // for debug purpose
    vp_utils::vp_analysis_board board({file_src_0, rtsp_src_1, udp_src_2});
    board.display();
}