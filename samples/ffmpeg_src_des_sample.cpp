//
// Created by jin_li on 2024/5/17.
//

#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_yunet_face_detector_node.h"
#include "../nodes/infers/vp_sface_feature_encoder_node.h"
#include "../nodes/osd/vp_face_osd_node_v2.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/vp_ffmpeg_des_node.h"
#include "../nodes/vp_ffmpeg_src_node.h"
#include "../nodes/vp_rtmp_des_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"

int main() {
    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_LOGGER_INIT();

    // create nodes
    auto ffmpeg_src_0 = std::make_shared<vp_nodes::vp_ffmpeg_src_node>("ffmpeg_src_0", 0, "/root/vp_data/test_video/face.mp4");
    auto yunet_face_detector_0 = std::make_shared<vp_nodes::vp_yunet_face_detector_node>("yunet_face_detector_0",
                                                                                         "/root/vp_data/face/face_detection_yunet_2022mar.onnx");
    auto sface_face_encoder_0 = std::make_shared<vp_nodes::vp_sface_feature_encoder_node>("sface_face_encoder_0",
                                                                                          "/root/vp_data/face/face_recognition_sface_2021dec.onnx");
    auto osd_0 = std::make_shared<vp_nodes::vp_face_osd_node_v2>("osd_0");
    auto ffmpeg_des_0 = std::make_shared<vp_nodes::vp_ffmpeg_des_node>("ffmpeg_des_0", 0,
                                                                            "rtmp://localhost:11935/videoPipe/test1",
                                                                            1280, 842, AV_PIX_FMT_BGR24, 1280, 842,
                                                                            AV_PIX_FMT_YUV420P);

    // construct pipeline
    yunet_face_detector_0->attach_to({ffmpeg_src_0});
    sface_face_encoder_0->attach_to({yunet_face_detector_0});
    osd_0->attach_to({sface_face_encoder_0});
    ffmpeg_des_0->attach_to({osd_0});

    ffmpeg_src_0->start();

    std::string wait;
    std::getline(std::cin, wait);
    ffmpeg_src_0->detach_recursively();

}
