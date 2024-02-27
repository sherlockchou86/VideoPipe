#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_trt_vehicle_detector.h"
#include "../nodes/infers/vp_trt_vehicle_color_classifier.h"
#include "../nodes/track/vp_sort_track_node.h"
#include "../nodes/osd/vp_osd_node.h"
#include "../nodes/broker/vp_json_console_broker_node.h"
#include "../nodes/vp_split_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## dynamic_pipeline_sample ##
* 1. insert and remove channels(SRC & DES nodes) to/from existing pipeline.
* 2. insert and remove MID nodes to/from existing pipeline.
* 3. all nodes destroy and process exit normally after user press enter from console.
* 4. no need to stop the pipeline and all operations in multi-threads, it works with hot-plug mode.
*/

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./vp_data/test_video/vehicle_stop.mp4", 0.4);
    auto trt_vehicle_detector = std::make_shared<vp_nodes::vp_trt_vehicle_detector>("trt_vehicle_detector", "./vp_data/models/trt/vehicle/vehicle_v8.5.trt");
    auto osd = std::make_shared<vp_nodes::vp_osd_node>("osd", "./vp_data/font/NotoSansCJKsc-Medium.otf");
    auto split = std::make_shared<vp_nodes::vp_split_node>("split", true);
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);

    // construct pipeline the first time
    trt_vehicle_detector->attach_to({file_src_0});
    osd->attach_to({trt_vehicle_detector});
    split->attach_to({osd});
    screen_des_0->attach_to({split});

    // start pipeline
    file_src_0->start();
    // visualize pipeline for debug
    std::vector<std::shared_ptr<vp_nodes::vp_node>> src_nodes = {file_src_0};
    vp_utils::vp_analysis_board board(src_nodes);
    board.display(1, false);  // no block
    /* the original format of pipeline is:
       file_src_0 -> trt_vehicle_detector -> osd -> split -> screen_des_0
    */

    // simulation function for dynamically operating on piepline 
    bool exit = false;
    auto simulate_run = [&]() {
        while (!exit) {    
            // 1. wait for 5 seconds then insert the 2nd channel(input and output) to pipeline
            std::this_thread::sleep_for(std::chrono::seconds(5));
            auto file_src_1 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_1", 1, "./vp_data/test_video/vehicle_stop.mp4", 0.4);
            auto screen_des_1 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_1", 1);
            trt_vehicle_detector->attach_to({file_src_1});
            screen_des_1->attach_to({split});
            // start 2nd channel
            file_src_1->start();
            // reload board using 2 SRC nodes
            src_nodes.push_back(file_src_1);
            board.reload(src_nodes);
            /* now the format of pipeline is:
            file_src_0 \                                         / screen_des_0
                        -> trt_vehicle_detector -> osd -> split -> 
            file_src_1 /                                         \ screen_des_1
            */
            
            // 2. wait for 5 seconds then insert the 3rd channel(input and output) to pipeline
            std::this_thread::sleep_for(std::chrono::seconds(5));
            auto file_src_2 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_2", 2, "./vp_data/test_video/vehicle_stop.mp4", 0.4);
            auto screen_des_2 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_2", 2);
            trt_vehicle_detector->attach_to({file_src_2});
            screen_des_2->attach_to({split});
            // start 3rd channel
            file_src_2->start();
            // reload board using 3 SRC nodes
            src_nodes.push_back(file_src_2);
            board.reload(src_nodes);
            /* now the format of pipeline is:
            file_src_0 \                                        / screen_des_0
            file_src_1 -> trt_vehicle_detector -> osd -> split -> screen_des_1
            file_src_2 /                                        \ screen_des_2
            */

            // 3. wait for 5 seconds then remove the DES node(output) of 3rd channel from pipeline
            std::this_thread::sleep_for(std::chrono::seconds(5));
            screen_des_2->detach();  // call detach() since there is only 1 previous node for screen_des_2
            screen_des_2 = nullptr;  // force call destructor immediately
            // reload board using previous SRC nodes
            board.reload();
            /* now the format of pipeline is:
            file_src_0 \                                        / screen_des_0
            file_src_1 -> trt_vehicle_detector -> osd -> split -> screen_des_1
            file_src_2 /
            */

            // 4. wait for 5 seconds then remove the SRC node(input) of 3rd channel from pipeline
            std::this_thread::sleep_for(std::chrono::seconds(5));
            trt_vehicle_detector->detach_from({"file_src_2"}); // call detach_from(...) since there are many previous nodes for trt_vehicle_detector
            file_src_2->stop();   // call stop() on SRC node which is not reused later
            file_src_2 = nullptr; // force call destructor immediately
            // reload board using 2 SRC nodes
            src_nodes.pop_back();
            board.reload(src_nodes);
            /* now the format of pipeline is:
            file_src_0 \                                         / screen_des_0
                        -> trt_vehicle_detector -> osd -> split -> 
            file_src_1 /                                         \ screen_des_1
            */
            
            // 5. wait for 5 seconds then insert a secondary classifier node, track node and broker node into pipeline
            std::this_thread::sleep_for(std::chrono::seconds(5));
            auto trt_color_cls = std::make_shared<vp_nodes::vp_trt_vehicle_color_classifier>("trt_color_cls", "./vp_data/models/trt/vehicle/vehicle_color_v8.5.trt", std::vector<int>{0, 1, 2});
            auto track = std::make_shared<vp_nodes::vp_sort_track_node>("track");
            auto console_broker = std::make_shared<vp_nodes::vp_json_console_broker_node>("console_broker");
            osd->detach();                      // first detach osd node from pipeline
            osd->attach_to({console_broker});                  // then attach osd node to broker node
            console_broker->attach_to({track});                // attach broker node to track node
            track->attach_to({trt_color_cls});                 // attach track node to classifier node 
            trt_color_cls->attach_to({trt_vehicle_detector});  // attach classifier node into pipeline
            // reload board using previous SRC nodes
            board.reload();
            /* now the format of pipeline is:
            file_src_0 \                                                                                    / screen_des_0
                        -> trt_vehicle_detector -> trt_color_cls -> track -> console_broker -> osd -> split -> 
            file_src_1 /                                                                                    \ screen_des_1
            */

            // 6. wait for 10 seconds then remove the 2nd channel (both SRC node and DES node) from pipeline
            std::this_thread::sleep_for(std::chrono::seconds(10));
            screen_des_1->detach();  // call detach() since there is only 1 previous node for screen_des_1
            trt_vehicle_detector->detach_from({"file_src_1"}); // call detach_from(...) since there are many previous nodes for trt_vehicle_detector
            file_src_1->stop(); // call stop() on SRC node which is not reused later
            screen_des_1 = nullptr; // force call destructor immediately
            file_src_1 = nullptr;   // force call destructor immediately
            // reload board using 1 SRC nodes
            src_nodes.pop_back();
            board.reload(src_nodes);
            /* now the format of pipeline is:
            file_src_0 -> trt_vehicle_detector -> trt_color_cls -> track -> console_broker -> osd -> split -> screen_des_0
            */

            // 7. wait for 10 seconds then remove classifier node, track node and broker node from pipeline
            std::this_thread::sleep_for(std::chrono::seconds(10));
            osd->detach();
            console_broker->detach();
            track->detach();
            trt_color_cls->detach();
            osd->attach_to({trt_vehicle_detector});  // relink osd and trt_vehicle_detector

            console_broker = nullptr;
            track = nullptr;
            trt_color_cls = nullptr;
            // reload board using previous SRC nodes
            board.reload();
            /* now the format of pipeline is:
            file_src_0 -> trt_vehicle_detector -> osd -> split -> screen_des_0
            */
        }
    }; 

    // start simulation thread
    std::thread simulator(simulate_run);

    // enter to exit
    std::string input;
    std::getline(std::cin, input);
    exit = true;
    simulator.join();

    // split pipeline into single nodes before process exit
    for (auto& n: src_nodes) {
        n->detach_recursively();
    }
    // pipeline destroyed and process exit normally
}