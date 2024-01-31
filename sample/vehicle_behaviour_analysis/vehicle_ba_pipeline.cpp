#include "../VP.h"

#include "../../nodes/vp_file_src_node.h"
#include "../../nodes/infers/vp_trt_vehicle_detector.h"
#include "../../nodes/infers/vp_trt_vehicle_color_classifier.h"
#include "../../nodes/infers/vp_trt_vehicle_type_classifier.h"
#include "../../nodes/track/vp_sort_track_node.h"
#include "../../nodes/record/vp_record_node.h"
#include "../../nodes/ba/vp_ba_crossline_node.h"
#include "../../nodes/broker/vp_ba_socket_broker_node.h"
#include "../../nodes/osd/vp_ba_crossline_osd_node.h"
#include "../../nodes/ba/vp_ba_stop_node.h"
#include "../../nodes/osd/vp_ba_stop_osd_node.h"
#include "../../nodes/vp_split_node.h"
#include "../../nodes/vp_placeholder_node.h"
#include "../../nodes/vp_screen_des_node.h"
#include "../../nodes/vp_rtmp_des_node.h"
#include "../../utils/analysis_board/vp_analysis_board.h"

/*
* ## vehicle_ba_sample ##
* count and stop check for vehicles using VideoPipe and send results via socket.
*/

#if vehicle_ba_sample

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "../test_video/vehicle_count.mp4", 0.5);
    auto file_src_1 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_1", 1, "../test_video/vehicle_stop2.mp4", 0.75);
    auto trt_vehicle_detector = std::make_shared<vp_nodes::vp_trt_vehicle_detector>("trt_detector", "../models/trt/vehicle/vehicle_v8.5.trt");
    auto trt_vehicle_color_classifier = std::make_shared<vp_nodes::vp_trt_vehicle_color_classifier>("trt_color_cls", "../models/trt/vehicle/vehicle_color_v8.5.trt", std::vector<int>{0, 1, 2}, 20, 20);
    auto trt_vehicle_type_classifier = std::make_shared<vp_nodes::vp_trt_vehicle_type_classifier>("trt_type_cls", "../models/trt/vehicle/vehicle_type_v8.5.trt", std::vector<int>{0, 1, 2}, 20, 20);
    auto tracker = std::make_shared<vp_nodes::vp_sort_track_node>("tracker");
    
    // generate config data (line) for ba crossline node
    vp_objects::vp_point start(0, 350);  // change to proper value
    vp_objects::vp_point end(960, 350);  // change to proper value
    std::map<int, vp_objects::vp_line> lines = {{0, vp_objects::vp_line(start, end)},  // channel0 -> line
                                                {1, vp_objects::vp_line(start, end)}}; // channel1 -> line
    auto ba_crossline = std::make_shared<vp_nodes::vp_ba_crossline_node>("ba_crossline", lines);
    
    // generate config data (region) for ba stop node
    std::map<int, std::vector<vp_objects::vp_point>> regions = {
        // ba stop node do not work on channel 0 since no config data for channel 0
        {1, std::vector<vp_objects::vp_point>{vp_objects::vp_point(20, 30), vp_objects::vp_point(1000, 40), vp_objects::vp_point(1000, 600), vp_objects::vp_point(10, 600)}}   // channel1 -> region
    };
    auto ba_stop = std::make_shared<vp_nodes::vp_ba_stop_node>("ba_stop", regions);

    auto crossline_osd = std::make_shared<vp_nodes::vp_ba_crossline_osd_node>("crossline_osd");
    auto stop_osd = std::make_shared<vp_nodes::vp_ba_stop_osd_node>("stop_osd");
    auto recorder = std::make_shared<vp_nodes::vp_record_node>("recorder", "./static/record_videos", "./static/record_images", vp_objects::vp_size(), true, 3, 5, false);
    auto ba_broker = std::make_shared<vp_nodes::vp_ba_socket_broker_node>("ba_broker", "192.168.77.68", 7777);
    
    auto split = std::make_shared<vp_nodes::vp_split_node>("split", true);
    auto placeholder_0 = std::make_shared<vp_nodes::vp_placeholder_node>("placeholder_0");
    auto placeholder_1 = std::make_shared<vp_nodes::vp_placeholder_node>("placeholder_1");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);
    auto rtmp_des_0 = std::make_shared<vp_nodes::vp_rtmp_des_node>("rtmp_des_0", 0, "rtmp://192.168.77.60/live/vehicle_ba_sample");
    auto screen_des_1 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_1", 1);
    auto rtmp_des_1 = std::make_shared<vp_nodes::vp_rtmp_des_node>("rtmp_des_1", 1, "rtmp://192.168.77.60/live/vehicle_ba_sample");
    
    // construct pipeline
    trt_vehicle_detector->attach_to({file_src_0, file_src_1});
    trt_vehicle_color_classifier->attach_to({trt_vehicle_detector});
    trt_vehicle_type_classifier->attach_to({trt_vehicle_color_classifier});
    tracker->attach_to({trt_vehicle_type_classifier});
    ba_crossline->attach_to({tracker});
    ba_stop->attach_to({ba_crossline});
    ba_broker->attach_to({ba_stop});
    crossline_osd->attach_to({ba_broker});
    stop_osd->attach_to({crossline_osd});
    recorder->attach_to({stop_osd});
    split->attach_to({recorder});

    // manually split into 2 branches using vp_split_node, put a placeholder behind since each branch will split into 2 branches again
    placeholder_0->attach_to({split});
    placeholder_1->attach_to({split});

    // automatically split into 2 branches
    screen_des_0->attach_to({placeholder_0});
    rtmp_des_0->attach_to({placeholder_0});
    screen_des_1->attach_to({placeholder_1});
    rtmp_des_1->attach_to({placeholder_1});

    // start pipeline
    file_src_0->start();
    file_src_1->start();

    // visualize pipeline for debug
    vp_utils::vp_analysis_board board({file_src_0, file_src_1});
    board.display();
}

#endif