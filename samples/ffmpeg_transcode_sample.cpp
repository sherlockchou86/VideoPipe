#include "../nodes/ffio/vp_ff_src_node.h"
#include "../nodes/infers/vp_yunet_face_detector_node.h"
#include "../nodes/infers/vp_sface_feature_encoder_node.h"
#include "../nodes/osd/vp_face_osd_node_v2.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/ffio/vp_ff_des_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## ffmpeg_transcode_sample ##
* transcoding in parallel, modify codec & resolution & bitrate & add watermask in picture.
*/

int main() {
    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto ff_src_0 = std::make_shared<vp_nodes::vp_ff_src_node>("ff_src_0", 0, "rtmp://192.168.77.196/live/1000", "h264", 0.5);
    auto ff_des_0 = std::make_shared<vp_nodes::vp_ff_des_node>("ff_des_0", 0, "rtmp://192.168.77.196/live/0");

    auto ff_src_1 = std::make_shared<vp_nodes::vp_ff_src_node>("ff_src_1", 0, "rtmp://192.168.77.196/live/1000", "h264", 0.5);
    auto ff_des_1 = std::make_shared<vp_nodes::vp_ff_des_node>("ff_des_1", 0, "rtmp://192.168.77.196/live/1");

    auto ff_src_2 = std::make_shared<vp_nodes::vp_ff_src_node>("ff_src_2", 0, "rtsp://192.168.77.213/live/mainstream", "h264", 0.5);
    auto ff_des_2 = std::make_shared<vp_nodes::vp_ff_des_node>("ff_des_2", 0, "rtmp://192.168.77.196/live/2");

    auto ff_src_3 = std::make_shared<vp_nodes::vp_ff_src_node>("ff_src_3", 0, "rtsp://192.168.77.213/live/mainstream", "h264", 0.5);
    auto ff_des_3 = std::make_shared<vp_nodes::vp_ff_des_node>("ff_des_3", 0, "rtmp://192.168.77.196/live/3");

    // construct pipeline
    ff_des_0->attach_to({ff_src_0});
    ff_des_1->attach_to({ff_src_1});
    ff_des_2->attach_to({ff_src_2});
    ff_des_3->attach_to({ff_src_3});

    // start pipelines
    ff_src_0->start();
    ff_src_1->start();
    ff_src_2->start();
    ff_src_3->start();

    // for debug purpose
    vp_utils::vp_analysis_board board({ff_src_0, ff_src_1, ff_src_2, ff_src_3});
    board.display(1, false);

    std::string wait;
    std::getline(std::cin, wait);
    ff_src_0->detach_recursively();
    ff_src_1->detach_recursively();
    ff_src_2->detach_recursively();
    ff_src_3->detach_recursively();
}