

#include "VP.h"

#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_yunet_face_detector_node.h"
#include "../nodes/infers/vp_sface_feature_encoder_node.h"
#include "../nodes/track/vp_sort_track_node.h"
#include "../nodes/osd/vp_face_osd_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/vp_rtmp_des_node.h"
#include "../nodes/vp_split_node.h"
#include "../nodes/record/vp_record_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## record sample ##
* show how to use vp_record_node to record image and video.
* NOTE:
* the recording signal in this demo is triggered by users outside pipe (via calling vp_src_node::record_video_manually or vp_src_node::record_image_manually)
* in product situations, recording signal is triggered inside pipe automatically.
*/

#if record_sample

int main() {
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_SET_LOG_TO_CONSOLE(false);   // need interact on console
    VP_LOGGER_INIT();

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./test_video/10.mp4", 0.6);
    auto file_src_1 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_1", 1, "./test_video/9.mp4", 0.6);
    auto yunet_face_detector = std::make_shared<vp_nodes::vp_yunet_face_detector_node>("yunet_face_detector_0", "./models/face/face_detection_yunet_2022mar.onnx");
    auto sface_face_encoder = std::make_shared<vp_nodes::vp_sface_feature_encoder_node>("sface_face_encoder_0", "./models/face/face_recognition_sface_2021dec.onnx");
    auto track = std::make_shared<vp_nodes::vp_sort_track_node>("track", vp_nodes::vp_track_for::FACE);
    auto osd = std::make_shared<vp_nodes::vp_face_osd_node>("osd");    
    auto recorder = std::make_shared<vp_nodes::vp_record_node>("recorder", "./record", "./record");

    auto split = std::make_shared<vp_nodes::vp_split_node>("split", true);  // split by channel index
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);
    auto screen_des_1 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_1", 1);

    // construct pipeline
    yunet_face_detector->attach_to({file_src_0, file_src_1});
    sface_face_encoder->attach_to({yunet_face_detector});
    track->attach_to({sface_face_encoder});
    osd->attach_to({track});
    recorder->attach_to({osd});
    split->attach_to({recorder});
    // split by vp_split_node
    screen_des_0->attach_to({split});
    screen_des_1->attach_to({split});

    /*
    * set hookers for vp_record_node when task compeleted
    */
    // define hooker 
    auto record_hooker = [](int channel, vp_nodes::vp_record_info record_info) {
        auto record_type = record_info.record_type == vp_nodes::vp_record_type::IMAGE ? "image" : "video";

        std::cout << "channel:[" << channel << "] [" <<  record_type << "]" <<  " record task completed! full path: " << record_info.full_record_path << std::endl;
    };
    recorder->set_image_record_complete_hooker(record_hooker);
    recorder->set_video_record_complete_hooker(record_hooker);

    // start channels
    file_src_0->start();
    file_src_1->start();

    // for debug purpose
    std::vector<std::shared_ptr<vp_nodes::vp_node>> src_nodes_in_pipe{file_src_0, file_src_1};
    vp_utils::vp_analysis_board board(src_nodes_in_pipe);
    board.display(1, false);  // no block

    
    /* interact from console */
    /* no except check */
    std::string input;
    std::getline(std::cin, input);
    // input format: `image channel` or `video channel`, like `video 0` means start recording video at channel 0
    auto inputs = vp_utils::string_split(input, ' '); 
    while (inputs[0] != "quit") {
        // no except check
        auto command = inputs[0];
        auto index = std::stoi(inputs[1]);
        auto src_by_channel = std::dynamic_pointer_cast<vp_nodes::vp_src_node>(src_nodes_in_pipe[index]);
        if (command == "video") {
            src_by_channel->record_video_manually(true);   // debug api
            // or
            // src_by_channel->record_video_manually(true, 5);
            // src_by_channel->record_video_manually(false, 20);
        }
        else if (command == "image") {
            src_by_channel->record_image_manually();   // debug api
            // or
            // src_by_channel->record_image_manually(true);
        }
        else {
            std::cout << "invalid command!" << std::endl;
        }
        std::getline(std::cin, input);
        inputs = vp_utils::string_split(input, ' '); 
        if (inputs.size() != 2) {
             std::cout << "invalid input!" << std::endl;
             break;
        }
    }

    std::cout << "record sample exits..." << std::endl;
}

#endif