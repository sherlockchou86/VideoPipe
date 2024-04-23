#include "../nodes/vp_file_src_node.h"
#include "../nodes/vp_rtsp_src_node.h"
#include "../nodes/infers/vp_yunet_face_detector_node.h"
#include "../nodes/infers/vp_sface_feature_encoder_node.h"
#include "../nodes/osd/vp_face_osd_node_v2.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/vp_rtmp_des_node.h"
#include "../nodes/vp_fake_des_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## vp_test ##
* test anything for videopipe in this cpp.
*/

int main() {
    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto rtsp_src_0 = std::make_shared<vp_nodes::vp_rtsp_src_node>("rtsp_src_0", 0, "rtsp://192.168.77.193:8554/stream/main", 1, "avdec_h264", 1);
    auto fake_des_0 = std::make_shared<vp_nodes::vp_fake_des_node>("fake_des_0", 0);
    auto rtsp_src_1 = std::make_shared<vp_nodes::vp_rtsp_src_node>("rtsp_src_1", 1, "rtsp://192.168.77.193:8554/stream/main", 1, "avdec_h264", 1);
    auto fake_des_1 = std::make_shared<vp_nodes::vp_fake_des_node>("fake_des_1", 1);
    auto rtsp_src_2 = std::make_shared<vp_nodes::vp_rtsp_src_node>("rtsp_src_2", 2, "rtsp://192.168.77.193:8554/stream/main", 1, "avdec_h264", 1);
    auto fake_des_2 = std::make_shared<vp_nodes::vp_fake_des_node>("fake_des_2", 2);
    auto rtsp_src_3 = std::make_shared<vp_nodes::vp_rtsp_src_node>("rtsp_src_3", 3, "rtsp://192.168.77.193:8554/stream/main", 1, "avdec_h264", 1);
    auto fake_des_3 = std::make_shared<vp_nodes::vp_fake_des_node>("fake_des_3", 3);
    auto rtsp_src_4 = std::make_shared<vp_nodes::vp_rtsp_src_node>("rtsp_src_4", 4, "rtsp://192.168.77.193:8554/stream/main", 1, "avdec_h264", 1);
    auto fake_des_4 = std::make_shared<vp_nodes::vp_fake_des_node>("fake_des_4", 4);

    // construct pipeline
    fake_des_0->attach_to({rtsp_src_0});
    fake_des_1->attach_to({rtsp_src_1});
    fake_des_2->attach_to({rtsp_src_2});
    fake_des_3->attach_to({rtsp_src_3});
    fake_des_4->attach_to({rtsp_src_4});

    // start
    rtsp_src_0->start();
    rtsp_src_1->start();
    rtsp_src_2->start();
    rtsp_src_3->start();
    rtsp_src_4->start();

    // for debug purpose
    vp_utils::vp_analysis_board board({rtsp_src_0, rtsp_src_1, rtsp_src_2, rtsp_src_3, rtsp_src_4});
    board.display(1, false);

    std::string wait;
    std::getline(std::cin, wait);
    rtsp_src_0->detach_recursively();
    rtsp_src_1->detach_recursively();
    rtsp_src_2->detach_recursively();
    rtsp_src_3->detach_recursively();
    rtsp_src_4->detach_recursively();
}