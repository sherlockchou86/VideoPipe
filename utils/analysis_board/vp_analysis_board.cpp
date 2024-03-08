

#include "vp_analysis_board.h"
#include "../vp_pipe_checker.h"

namespace vp_utils {
    vp_analysis_board::vp_analysis_board(std::vector<std::shared_ptr<vp_nodes::vp_node>> src_nodes_in_pipe):
        src_nodes_in_pipe(src_nodes_in_pipe) {   
        init();
    }
    
    vp_analysis_board::~vp_analysis_board() {
        // set alive to false and wait threads exits
        alive = false;
        if (display_th.joinable()) {
            display_th.join();
        }
        if (rtmp_th.joinable()) {
            rtmp_th.join();
        }
    }

    void vp_analysis_board::init() {
        src_nodes_on_screen.clear();
        des_nodes_on_screen.clear();

        // check pipe
        vp_pipe_checker pipe_checker;
        pipe_checker(src_nodes_in_pipe);

        // layers number and max nodes number of all layers
        pipe_width = pipe_checker.pipe_width();
        pipe_height = pipe_checker.pipe_height();

        // calculate the w and h of canvas
        canvas_width = pipe_width * node_width + (pipe_width - 1) * node_gap_horizontal + 2 * canvas_gap_horizontal;
        canvas_height = pipe_height * node_height + (pipe_height - 1) * node_gap_vertical + 2 * canvas_gap_vertical; 

        // create canvas Mat and initialize it with white
        bg_canvas.create(canvas_height, canvas_width, CV_8UC3);
        bg_canvas = cv::Scalar(255, 255, 255);

        // map recursively
        map_nodes(src_nodes_on_screen, 1);

        // render static parts starting with 1st layer
        render_layer(src_nodes_on_screen, bg_canvas);

        // save to local by default
        save(board_title + ".png");
    }
    void vp_analysis_board::reload(std::vector<std::shared_ptr<vp_nodes::vp_node>> new_src_nodes_in_pipe) {
        std::lock_guard<std::mutex> guard(reload_lock);
        if (new_src_nodes_in_pipe.size() != 0) {
            this->src_nodes_in_pipe = new_src_nodes_in_pipe;
        }
        init();
    }

    void vp_analysis_board::save(std::string path) {
        cv::imwrite(path, bg_canvas);
    }

    void vp_analysis_board::display(int interval, bool block) {
        assert(interval > 0);
        if (displaying) {
            return;
        }
        
        auto display_func = [&, interval](){
            while (alive) {
                auto loop_start = std::chrono::system_clock::now();
                {
                    std::lock_guard<std::mutex> guard(reload_lock);  // in case it reloading
                    // deep copy the static background
                    cv::Mat mat_to_display = bg_canvas.clone();
                    // render dynamic parts starting with 1 st layer
                    render_layer(src_nodes_on_screen, mat_to_display, false);
                    cv::imshow(board_title, mat_to_display);
                }

                // calculate the time need wait for
                auto loop_cost = std::chrono::system_clock::now() - loop_start;
                auto wait_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::seconds(interval) - loop_cost);

                cv::waitKey(wait_time.count());
            }};
        displaying = true;
        display_th = std::thread(display_func);
        if (block) {
            display_th.join();
        }
    }

    void vp_analysis_board::push_rtmp(std::string rtmp, int bitrate) {
        if (displaying) {
            return;
        }
        auto fps = 10;
        auto rtmp_url = vp_utils::string_format(gst_template, bitrate, rtmp.c_str());
        // 10 fps by default
        assert(rtmp_writer.open(rtmp_url, cv::CAP_GSTREAMER, fps, {bg_canvas.cols, bg_canvas.rows}));

        auto display_func = [&, fps](){
            while (alive) {
                auto loop_start = std::chrono::system_clock::now();
                {
                    std::lock_guard<std::mutex> guard(reload_lock); // in case it reloading
                    // deep copy the static background
                    cv::Mat mat_to_display = bg_canvas.clone();
                    // render dynamic parts starting with 1 st layer
                    render_layer(src_nodes_on_screen, mat_to_display, false);
                    rtmp_writer.write(mat_to_display);
                }

                // calculate the time need wait for
                auto loop_cost = std::chrono::system_clock::now() - loop_start;
                auto wait_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::milliseconds(1000 / fps) - loop_cost);

                std::this_thread::sleep_for(wait_time);
            }};
        displaying = true;
        rtmp_th = std::thread(display_func);
    }

    void vp_analysis_board::render_layer(std::vector<std::shared_ptr<vp_node_on_screen>> nodes_in_layer, cv::Mat& canvas, bool static_parts) {
        std::vector<std::shared_ptr<vp_node_on_screen>> nodes_in_next_layer;
        for(auto& i : nodes_in_layer) {
            if (static_parts) {             
                i->render_static_parts(canvas);
            }
            else {
                i->render_dynamic_parts(canvas);
            }
            auto n = i->get_next_nodes_on_screen();
            nodes_in_next_layer.insert(nodes_in_next_layer.end(), n.begin(), n.end());
        }
        
        if (nodes_in_next_layer.size() > 0) {
            bool all_the_same = true;
            for(auto & i: nodes_in_next_layer) {
                if (i != nodes_in_next_layer[0]) {
                    all_the_same = false;
                    break;
                }
            }
            // just keep the first one if all the next nodes are the same
            if (all_the_same) {
                nodes_in_next_layer.erase(nodes_in_next_layer.begin() + 1, nodes_in_next_layer.end()); 
            }

            render_layer(nodes_in_next_layer, canvas, static_parts);
        }
        else { // recursion end
            
            /* global drawing */

            // draw layer index at the bottom of canvas
            if (static_parts) {
                for (int i = 0; i < pipe_width; i++) {
                    /* code */
                    cv::putText(canvas, "layer_" + std::to_string(i + 1), cv::Point(canvas_gap_horizontal + (node_width + node_gap_horizontal) * i, canvas_height - int(canvas_gap_vertical / 3)), 1, 1, cv::Scalar(255, 0, 0));
                }    
            }

            // refresh time at the top left of canvas
            if (!static_parts) {
                auto time = vp_utils::time_format(NOW, "<hour>:<min>:<sec>");
                cv::putText(canvas, time, cv::Point(20, 20), 1, 1, cv::Scalar(255, 0, 0));
            }   
        }   
    }

    void vp_analysis_board::map_nodes(std::vector<std::shared_ptr<vp_node_on_screen>> nodes_in_layer, int layer) {
        if (layer == 1) {  
            // here nodes_in_layer is empty
            auto num_src_nodes_in_pipe = src_nodes_in_pipe.size();
            
            auto base_left = layer_base_left_cal(layer - 1);
            auto base_top = layer_base_top_cal(num_src_nodes_in_pipe);

            // map nodes at 1st layer in memory to screen
            for (int i = 0; i < num_src_nodes_in_pipe; i++) {
                auto node_left = base_left;
                auto node_top = base_top + i * (node_height + node_gap_vertical);

                auto node_on_screen = std::make_shared<vp_node_on_screen>(src_nodes_in_pipe[i], vp_objects::vp_rect(node_left, node_top, node_width, node_height), layer);
                src_nodes_on_screen.push_back(node_on_screen);
            }

            map_nodes(src_nodes_on_screen, layer + 1);
        }
        else {
            std::vector<std::shared_ptr<vp_nodes::vp_node>> all_nodes_in_next_layer;
            for(auto &i: nodes_in_layer) {
                auto next_nodes = i->get_orginal_node()->next_nodes();
                all_nodes_in_next_layer.insert(all_nodes_in_next_layer.end(), next_nodes.begin(), next_nodes.end());
            }
            if (all_nodes_in_next_layer.size() > 0) {       
                bool all_the_same = true;
                for(auto & i: all_nodes_in_next_layer) {
                    if (i != all_nodes_in_next_layer[0]) {
                        all_the_same = false;
                        break;
                    }
                }
                // just keep the first one if all the next nodes are the same
                if (all_the_same) {
                    all_nodes_in_next_layer.erase(all_nodes_in_next_layer.begin() + 1, all_nodes_in_next_layer.end()); 
                }
            
                auto num_all_nodes_in_next_layer = all_nodes_in_next_layer.size();
                auto base_left = layer_base_left_cal(layer - 1);
                auto base_top = layer_base_top_cal(num_all_nodes_in_next_layer);

                auto index = 0;
                std::shared_ptr<vp_node_on_screen> node_on_screen = nullptr;
                std::vector<std::shared_ptr<vp_node_on_screen>> nodes_in_next_layer;
                for(int i = 0; i < nodes_in_layer.size(); i++) {
                    auto node_left = base_left;
                    auto next_nodes_in_pipe = nodes_in_layer[i]->get_orginal_node()->next_nodes();
                    for (int j = 0; j < next_nodes_in_pipe.size(); j++)
                    {
                        auto node_top = base_top +  index * (node_height + node_gap_vertical);
                        if (!all_the_same || node_on_screen == nullptr) {               
                            node_on_screen = std::make_shared<vp_node_on_screen>(next_nodes_in_pipe[j], vp_objects::vp_rect(node_left, node_top, node_width, node_height), layer);
                        }
                        nodes_in_layer[i]->get_next_nodes_on_screen().push_back(node_on_screen);

                        if (!all_the_same || nodes_in_next_layer.empty()) {                          
                            nodes_in_next_layer.push_back(node_on_screen);
                        }
                        index++;
                    }
                }
                // next layer
                map_nodes(nodes_in_next_layer, layer + 1);
            }
            else {
                // cache the last layer
                des_nodes_on_screen = nodes_in_layer;
            } // recursion end
        }
    }
}