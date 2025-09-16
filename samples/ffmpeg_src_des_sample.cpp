#include "../nodes/ffio/vp_ff_src_node.h"
#include "../nodes/infers/vp_yunet_face_detector_node.h"
#include "../nodes/infers/vp_sface_feature_encoder_node.h"
#include "../nodes/osd/vp_face_osd_node_v2.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/ffio/vp_ff_des_node.h"
#include "../nodes/vp_rtmp_des_node.h"
#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## ffmpeg_src_des_sample ##
* reading & pushing stream based on ffmpeg (soft decode and encode using CPUs).
*/

int main() {
    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    /*
       uri for vp_ff_src_node:
       1. rtmp://192.168.77.196/live/1000        --> reading rtmp live stream
       2. rtsp://192.168.77.213/live/mainstream  --> reading rtsp live stream
       3. ./vp_data/test_video/face.mp4          --> reading video file

       uri for vp_ff_des_node:
       1. rtmp://192.168.77.196/live/10000       --> pushing rtmp live stream
       2. ./output/records.mp4                   --> saving to video file
    */
    auto ff_src_0 = std::make_shared<vp_nodes::vp_ff_src_node>("ff_src_0", 0, "./vp_data/test_video/face.mp4", "h264", 0.6);
    auto yunet_face_detector_0 = std::make_shared<vp_nodes::vp_yunet_face_detector_node>("yunet_face_detector_0", "./vp_data/models/face/face_detection_yunet_2022mar.onnx");
    auto sface_face_encoder_0 = std::make_shared<vp_nodes::vp_sface_feature_encoder_node>("sface_face_encoder_0", "./vp_data/models/face/face_recognition_sface_2021dec.onnx");
    auto osd_0 = std::make_shared<vp_nodes::vp_face_osd_node_v2>("osd_0");
    auto ff_des_0 = std::make_shared<vp_nodes::vp_ff_des_node>("ff_des_0", 0, "rtmp://192.168.77.60/live/20000");
    //auto rtmp_des_0 = std::make_shared<vp_nodes::vp_rtmp_des_node>("rtmp_des_0", 0, "rtmp://192.168.77.196/live/20000");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);

    // construct pipeline
    yunet_face_detector_0->attach_to({ff_src_0});
    sface_face_encoder_0->attach_to({yunet_face_detector_0});
    osd_0->attach_to({sface_face_encoder_0});
    ff_des_0->attach_to({osd_0});
    screen_des_0->attach_to({osd_0});

    ff_src_0->start();

    // for debug purpose
    vp_utils::vp_analysis_board board({ff_src_0});
    board.display(1, false);

    std::string wait;
    std::getline(std::cin, wait);
    ff_src_0->detach_recursively();
}