#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_ppocr_text_detector_node.h"
#include "../nodes/osd/vp_text_osd_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/vp_rtmp_des_node.h"
#include "../utils/analysis_board/vp_analysis_board.h"
#include "../nodes/vp_file_des_node.h"

/*
* ## paddle infer sample ##
* ocr based on paddle (install paddle_inference first!)
* 1 video input and 2 outputs (screen, rtmp)
*/

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./vp_data/test_video/ocr.mp4", 0.4);
    auto ppocr_text_detector = std::make_shared<vp_nodes::vp_ppocr_text_detector_node>("ppocr_text_detector", "./vp_data/models/text/ppocr/ch_PP-OCRv3_det_infer","./vp_data/models/text/ppocr/ch_ppocr_mobile_v2.0_cls_infer","./vp_data/models/text/ppocr/ch_PP-OCRv3_rec_infer","./vp_data/models/text/ppocr/ppocr_keys_v1.txt");
    auto osd_0 = std::make_shared<vp_nodes::vp_text_osd_node>("osd_0", "./vp_data/font/NotoSansCJKsc-Medium.otf");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0, true);
    auto rtmp_des_0 = std::make_shared<vp_nodes::vp_rtmp_des_node>("rtmp_des_0", 0, "rtmp://192.168.77.60/live/10000");

    // construct pipeline
    ppocr_text_detector->attach_to({file_src_0});
    osd_0->attach_to({ppocr_text_detector});

    // split into 2 sub branches automatically
    screen_des_0->attach_to({osd_0});
    rtmp_des_0->attach_to({osd_0});

    // start pipeline
    file_src_0->start();

    // visualize pipeline for debug
    vp_utils::vp_analysis_board board({file_src_0});
    board.display();
}