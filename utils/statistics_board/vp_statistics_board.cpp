
#include <chrono>

#include "vp_statistics_board.h"
#include "../vp_pipe_checker.h"
#include "../vp_utils.h"

namespace vp_utils {
    
    vp_statistics_board::vp_statistics_board(std::vector<std::shared_ptr<vp_nodes::vp_node>> pipe_src_nodes)
        : pipe_src_nodes(pipe_src_nodes) {
            // check if the pipe legal or not
            vp_pipe_checker pipe_checker;
            pipe_checker(pipe_src_nodes);

            // layers number and max nodes number of all layers
            pipe_width = pipe_checker.pipe_width();
            pipe_height = pipe_checker.pipe_height();

            canvas_width = pipe_width * node_width + (pipe_width - 1) * node_gap_horizontal + 2 * canvas_gap_horizontal;
            canvas_height = pipe_height * node_height + (pipe_height - 1) * node_gap_vertical + 2 * canvas_gap_vertical;

            // initialize hookers, non-block!
            arriving_hooker = [this](std::string node_name, int queue_size, std::shared_ptr<vp_objects::vp_meta> meta) {
                this->arriving_hooker_storages[node_name].meta = meta;
                this->arriving_hooker_storages[node_name].queue_size = queue_size;
                this->arriving_hooker_storages[node_name].called_count_since_epoch_start++;
            };
            handling_hooker = [this](std::string node_name, int queue_size, std::shared_ptr<vp_objects::vp_meta> meta) {
                this->handling_hooker_storages[node_name].meta = meta;
                this->handling_hooker_storages[node_name].queue_size = queue_size;
                this->handling_hooker_storages[node_name].called_count_since_epoch_start++;
            };
            handled_hooker = [this](std::string node_name, int queue_size, std::shared_ptr<vp_objects::vp_meta> meta) {
                this->handled_hooker_storages[node_name].meta = meta;
                this->handled_hooker_storages[node_name].queue_size = queue_size;
                this->handled_hooker_storages[node_name].called_count_since_epoch_start++;
            };
            leaving_hooker = [this](std::string node_name, int queue_size, std::shared_ptr<vp_objects::vp_meta> meta) {
                this->leaving_hooker_storages[node_name].meta = meta;
                this->leaving_hooker_storages[node_name].queue_size = queue_size;
                this->leaving_hooker_storages[node_name].called_count_since_epoch_start++;
            };

            stream_info_hooker = [this](std::string node_name, vp_nodes::vp_stream_info stream_info) {
                this->stream_info_hooker_storages[node_name] = stream_info;
            };
    }

    vp_statistics_board::~vp_statistics_board() {

    }

    void vp_statistics_board::display(int interval, bool block) {
        assert(interval > 0);
        render_static_parts();  
        fps_epoch = interval * 500;

        auto display_func = [&](){
            while (true) {
                auto loop_start = std::chrono::system_clock::now();

                // deep copy the static background
                cv::Mat mat_to_display = bg_canvas.clone();
                render_dynamic_parts(mat_to_display);
                cv::imshow("vp_statistics_board", mat_to_display);

                // calculate the time need wait for
                auto loop_cost = std::chrono::system_clock::now() - loop_start;
                auto wait_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::seconds(interval) - loop_cost);

                cv::waitKey(wait_time.count());
            }};
        
        display_th = std::thread(display_func);
        if (block) {
            display_th.join();
        }
    }

    void vp_statistics_board::save_graph(std::string path) {
        render_static_parts();
        // save to ..
    }    

    void vp_statistics_board::render_static_parts() {
        if (!bg_canvas.empty()) {
            return;
        }
        bg_canvas.create(canvas_height, canvas_width, CV_8UC3);
        bg_canvas = cv::Scalar(255, 255, 255);

        // start render 1st layer of pipe
        render_layer(pipe_src_nodes, 1);
    }

    void vp_statistics_board::render_dynamic_parts(cv::Mat& ouput) {
        /*
        * draw data from hookers' callbacks
        */
        auto fps_func = [&](std::pair<const std::string, vp_hooker_storage> *i, cv::Rect rect) {
            auto called_count = i->second.called_count_since_epoch_start;
            auto epoch_start = i->second.time_epoch_start;
            auto delta_sec = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - epoch_start);
            if (delta_sec.count() > fps_epoch)
            {
                int fps = round(called_count * 1000.0 / delta_sec.count());
                i->second.called_count_since_epoch_start = 0;
                i->second.time_epoch_start = std::chrono::system_clock::now();
                vp_utils::put_text_at_center_of_rect(ouput, 
                    std::to_string(fps), 
                    rect, true,
                    font_face, 1, cv::Scalar(26, 26, 139));
            }
        };

        // draw in_queue size of each node dynamically on canvas
        for(auto & i : arriving_hooker_storages) {
            // -1 means hooker is not activated at all maybe it is a src node.
            if (i.second.queue_size >= 0) { 
                vp_utils::put_text_at_center_of_rect(ouput, 
                                        std::to_string(i.second.queue_size), 
                                        cv::Rect(i.second.node_rect.x + 3, 
                                        i.second.node_rect.y + node_title_h / 2 + (node_height - node_title_h) / 2, node_queue_width  - 8, node_title_h - 10), true, font_face, 1, cv::Scalar(255, 0, 255));
                // fps
                fps_func(&i, cv::Rect(i.second.node_rect.x - node_queue_width / 3, 
                            i.second.node_rect.y + i.second.node_rect.height - node_queue_port_padding * 3 - node_queue_port_w_h * 3 / 2, 
                            node_queue_width * 2 / 3, 
                            node_queue_port_padding + node_queue_port_w_h));
            }   
        }
        // draw out_queue size of each node dynamically on canvas
        for(auto & i : leaving_hooker_storages) {
            // -1 means hooker is not activated at all maybe it is a des node.
            if (i.second.queue_size >= 0) { 
                vp_utils::put_text_at_center_of_rect(ouput, 
                                        std::to_string(i.second.queue_size), 
                                        cv::Rect(i.second.node_rect.x + node_width - node_queue_width + 3, 
                                        i.second.node_rect.y + node_title_h / 2 + (node_height - node_title_h) / 2, node_queue_width - 8, node_title_h - 10), true, font_face, 1, cv::Scalar(255, 0, 255));
                // fps
                fps_func(&i, cv::Rect(i.second.node_rect.x + i.second.node_rect.width - node_queue_width / 3, 
                            i.second.node_rect.y + node_title_h + node_queue_port_padding * 3 / 2 + node_queue_port_w_h, 
                            node_queue_width * 2 / 3, 
                            node_queue_port_padding  + node_queue_port_w_h));
            }   
        }
        //
        for (auto & i: handling_hooker_storages) {
            if (i.second.queue_size >= 0) {
                // fps
                fps_func(&i, cv::Rect(i.second.node_rect.x + node_queue_width - node_queue_width / 3, 
                            i.second.node_rect.y + node_title_h + node_queue_port_padding * 3 / 2 + node_queue_port_w_h, 
                            node_queue_width * 2 / 3, 
                            node_queue_port_padding + node_queue_port_w_h));
            }
        }
        //
        for (auto & i: handled_hooker_storages) {
            if (i.second.queue_size >= 0) {
                // fps
                fps_func(&i, cv::Rect(i.second.node_rect.x + i.second.node_rect.width - node_queue_width - node_queue_width / 3, 
                            i.second.node_rect.y + i.second.node_rect.height - node_queue_port_padding * 3 - node_queue_port_w_h * 3 / 2, 
                            node_queue_width * 2 / 3, 
                            node_queue_port_padding + node_queue_port_w_h));
            }
        }

        
        // stream info at src nodes
        for(auto & i : stream_info_hooker_storages) {
            auto node_left = leaving_hooker_storages[i.first].node_rect.x;
            auto node_top = leaving_hooker_storages[i.first].node_rect.y;
            vp_utils::put_text_at_center_of_rect(bg_canvas, i.second.uri, 
                                                cv::Rect(node_left - node_width * 3 / 4, node_top + node_title_h + node_queue_port_padding, node_width * 4 / 3, node_title_h * 2 / 3), 
                                                true, font_face, 1, cv::Scalar(), cv::Scalar(), cv::Scalar(255, 255, 255));
            vp_utils::put_text_at_center_of_rect(bg_canvas, "original_width: " + std::to_string(i.second.original_width),
                                                cv::Rect(node_left - node_width * 3 / 4, node_top + node_title_h * 5 / 3 + node_queue_port_padding * 2, node_width * 4 / 3, node_title_h * 2 / 3),
                                                true, font_face, 1, cv::Scalar(), cv::Scalar(), cv::Scalar(255, 255, 255));
            vp_utils::put_text_at_center_of_rect(bg_canvas, "original_height: " + std::to_string(i.second.original_height),
                                                cv::Rect(node_left - node_width * 3 / 4, node_top + node_title_h * 7 / 3 + node_queue_port_padding * 3, node_width * 4 / 3, node_title_h * 2 / 3),
                                                true, font_face, 1, cv::Scalar(), cv::Scalar(), cv::Scalar(255, 255, 255));      
            vp_utils::put_text_at_center_of_rect(bg_canvas, "fps: " + std::to_string(i.second.fps),
                                                cv::Rect(node_left - node_width * 3 / 4, node_top + node_title_h * 9 / 3 + node_queue_port_padding * 4, node_width * 4 / 3, node_title_h * 2 / 3),
                                                true, font_face, 1, cv::Scalar(), cv::Scalar(), cv::Scalar(255, 255, 255));      
        }
    }

    void vp_statistics_board::render_layer(std::vector<std::shared_ptr<vp_nodes::vp_node>> nodes_in_layer, int layer) {
        auto num_nodes = nodes_in_layer.size();
        
        // base value
        auto layer_base_left_cal = [=](int layer_index) {return canvas_gap_horizontal + layer_index * ( node_width + node_gap_horizontal);};
        auto layer_base_top_cal = [=](int num_nodes_in_layer) {return (canvas_height - (num_nodes_in_layer * node_height + (num_nodes_in_layer - 1) * node_gap_vertical)) / 2; };
        auto base_left = layer_base_left_cal(layer - 1);
        auto base_top = layer_base_top_cal(num_nodes);

        std::vector<std::shared_ptr<vp_nodes::vp_node>> all_next_nodes;

        // render current layer
        for (int i = 0; i < num_nodes; i++) {
            auto node_left = base_left;
            auto node_top = base_top + i * (node_height + node_gap_vertical);

            cv::rectangle(bg_canvas, cv::Rect(node_left, node_top, node_width, node_height), cv::Scalar(0, 0, 0), 1);
            // node_name
            vp_utils::put_text_at_center_of_rect(bg_canvas, nodes_in_layer[i]->node_name, cv::Rect(node_left, node_top + 1, node_width, node_title_h - 2), false, font_face);
            cv::line(bg_canvas, 
                    cv::Point(node_left, node_top + node_title_h), 
                    cv::Point(node_left + node_width - 1, node_top + node_title_h), 
                    cv::Scalar(0, 0, 0), 1);

            // draw in_queue for non-src nodes
            if (nodes_in_layer[i]->node_type() != vp_nodes::vp_node_type::SRC) {
                // connect line between in_queue and out_queue
                if (nodes_in_layer[i]->node_type() == vp_nodes::vp_node_type::MID) {
                    cv::line(bg_canvas, 
                            cv::Point(node_left + node_queue_width + node_queue_port_w_h, node_top + node_title_h + node_queue_port_padding + node_queue_port_w_h / 2), 
                            cv::Point(node_left + node_width / 2, node_top + node_title_h + node_queue_port_padding + node_queue_port_w_h / 2),
                            cv::Scalar(156, 156, 156), 1);
                    cv::line(bg_canvas, 
                            cv::Point(node_left + node_width / 2, node_top + node_title_h + node_queue_port_padding + node_queue_port_w_h / 2), 
                            cv::Point(node_left + node_width / 2, node_top + node_height - node_queue_port_padding - node_queue_port_w_h / 2),
                            cv::Scalar(156, 156, 156), 1);
                    cv::line(bg_canvas, 
                            cv::Point(node_left + node_width / 2, node_top + node_height - node_queue_port_padding - node_queue_port_w_h / 2), 
                            cv::Point(node_left + node_width - node_queue_width - node_queue_port_w_h, node_top + node_height - node_queue_port_padding - node_queue_port_w_h / 2),
                            cv::Scalar(156, 156, 156), 1);
                    std::vector<cv::Point> vertexs {cv::Point(node_left + node_width - node_queue_width - node_queue_port_w_h, node_top + node_height - node_queue_port_padding - node_queue_port_w_h / 2), 
                                                    cv::Point(node_left + node_width - node_queue_width - node_queue_port_w_h * 2, node_top + node_height - node_queue_port_padding - node_queue_port_w_h), 
                                                    cv::Point(node_left + node_width - node_queue_width - node_queue_port_w_h * 2, node_top + node_height - node_queue_port_padding)};
                    cv::fillPoly(bg_canvas, std::vector<std::vector<cv::Point>>{vertexs}, cv::Scalar(156, 156, 156));
                }
                else {
                    // DES
                    cv::line(bg_canvas, 
                            cv::Point(node_left + node_queue_width + node_queue_port_w_h, node_top + node_title_h + node_queue_port_padding + node_queue_port_w_h / 2), 
                            cv::Point(node_left + node_width / 2, node_top + node_title_h + node_queue_port_padding + node_queue_port_w_h / 2),
                            cv::Scalar(156, 156, 156), 1);
                    std::vector<cv::Point> vertexs {cv::Point(node_left + node_width / 2, node_top + node_title_h + node_queue_port_padding), 
                                                    cv::Point(node_left + node_width / 2, node_top + node_title_h + node_queue_port_padding + node_queue_port_w_h), 
                                                    cv::Point(node_left + node_width / 2 + node_queue_port_w_h, node_top+ node_title_h + node_queue_port_padding + node_queue_port_w_h / 2)};
                    cv::fillPoly(bg_canvas, std::vector<std::vector<cv::Point>>{vertexs}, cv::Scalar(156, 156, 156));
                }

                cv::line(bg_canvas, 
                        cv::Point(node_left + node_queue_width, node_top + node_title_h), 
                        cv::Point(node_left + node_queue_width, node_top + node_height - 1), cv::Scalar(0, 0, 0), 1);

                // in port
                cv::rectangle(bg_canvas, 
                                cv::Rect(node_left - node_queue_port_w_h + 1, node_top + node_height - node_queue_port_padding - node_queue_port_w_h, node_queue_port_w_h, node_queue_port_w_h), 
                                cv::Scalar(156, 156, 156), 1);

                // out port
                cv::rectangle(bg_canvas, 
                                cv::Rect(node_left + node_queue_width, node_top + node_title_h + node_queue_port_padding, node_queue_port_w_h, node_queue_port_w_h), 
                                cv::Scalar(156, 156, 156), 1);
            }
            // draw out_queue for non-des nodes
            if (nodes_in_layer[i]->node_type() != vp_nodes::vp_node_type::DES) {
                // connect line between in_queue and out_queue 
                if (nodes_in_layer[i]->node_type() == vp_nodes::vp_node_type::SRC) {
                    cv::line(bg_canvas, 
                            cv::Point(node_left + node_width / 2, node_top + node_height - node_queue_port_padding - node_queue_port_w_h / 2), 
                            cv::Point(node_left + node_width - node_queue_width - node_queue_port_w_h, node_top + node_height - node_queue_port_padding - node_queue_port_w_h / 2), 
                            cv::Scalar(156, 156, 156), 1);
                    
                    std::vector<cv::Point> vertexs {cv::Point(node_left + node_width - node_queue_width - node_queue_port_w_h, node_top + node_height - node_queue_port_padding - node_queue_port_w_h / 2), 
                                                    cv::Point(node_left + node_width - node_queue_width - node_queue_port_w_h * 2, node_top + node_height - node_queue_port_padding - node_queue_port_w_h), 
                                                    cv::Point(node_left + node_width - node_queue_width - node_queue_port_w_h * 2, node_top + node_height - node_queue_port_padding)};
                    cv::fillPoly(bg_canvas, std::vector<std::vector<cv::Point>>{vertexs}, cv::Scalar(156, 156, 156));
                    
                    // hook on src nodes for stream info
                    auto src_node_ptr = std::dynamic_pointer_cast<vp_nodes::vp_src_node>(nodes_in_layer[i]);
                    src_node_ptr->set_stream_info_hooker(stream_info_hooker);
                    stream_info_hooker_storages[src_node_ptr->node_name] = vp_nodes::vp_stream_info();
                }
                
                cv::line(bg_canvas, 
                        cv::Point(node_left + node_width - node_queue_width, node_top + node_title_h), 
                        cv::Point(node_left + node_width - node_queue_width, node_top + node_height - 1), 
                        cv::Scalar(0, 0, 0), 1);
                
                // in port
                cv::rectangle(bg_canvas, 
                                cv::Rect(node_left + node_width - node_queue_width - node_queue_port_w_h + 1, node_top + node_height - node_queue_port_padding - node_queue_port_w_h, node_queue_port_w_h, node_queue_port_w_h), 
                                cv::Scalar(156, 156, 156), 1);
                // out port
                cv::rectangle(bg_canvas,
                                cv::Rect(node_left + node_width -1 , node_top + node_title_h + node_queue_port_padding, node_queue_port_w_h, node_queue_port_w_h), 
                                cv::Scalar(156, 156, 156), 1);
            }
            
            // center
            /*
            cv::circle(bg_canvas, 
                        cv::Point(node_left + (node_width - node_queue_width) / 2, node_top + node_title_h + (node_height - node_title_h) / 2), 
                        node_handle_logic_radius, cv::Scalar(0, 0, 0), 1, cv::LINE_AA); */
            
            auto next_nodes = nodes_in_layer[i]->next_nodes();
            all_next_nodes.insert(all_next_nodes.end(), next_nodes.begin(), next_nodes.end());

            // by the way, hook the node and initialize the meta_hookers' storage.
            nodes_in_layer[i]->set_meta_arriving_hooker(arriving_hooker);
            arriving_hooker_storages[nodes_in_layer[i]->node_name] 
                = vp_hooker_storage {vp_objects::vp_rect{node_left, node_top, node_width, node_height}, -1, -1, std::chrono::system_clock::now(), nullptr};

            nodes_in_layer[i]->set_meta_handling_hooker(handling_hooker);
            handling_hooker_storages[nodes_in_layer[i]->node_name] 
                = vp_hooker_storage {vp_objects::vp_rect{node_left, node_top, node_width, node_height}, -1, -1, std::chrono::system_clock::now(), nullptr};

            nodes_in_layer[i]->set_meta_handled_hooker(handled_hooker);
            handled_hooker_storages[nodes_in_layer[i]->node_name] 
                = vp_hooker_storage {vp_objects::vp_rect{node_left, node_top, node_width, node_height}, -1, -1, std::chrono::system_clock::now(), nullptr};

            nodes_in_layer[i]->set_meta_leaving_hooker(leaving_hooker);
            leaving_hooker_storages[nodes_in_layer[i]->node_name] 
                = vp_hooker_storage {vp_objects::vp_rect{node_left, node_top, node_width, node_height}, -1, -1, std::chrono::system_clock::now(), nullptr};
        }

        // layer text
        vp_utils::put_text_at_center_of_rect(bg_canvas, "layer-" + std::to_string(layer), cv::Rect(base_left, canvas_height - canvas_gap_vertical * 2 / 3, node_width, node_title_h - 2), false, font_face, 1, cv::Scalar(255, 0, 0));

        if (all_next_nodes.size() != 0) {
            bool all_the_same = true;
            for(auto & i: all_next_nodes) {
                if (i != all_next_nodes[0])
                {
                    all_the_same = false;
                    break;
                }
            }
            // just keep the first one if all the next nodes are the same
            if (all_the_same) {
                all_next_nodes.erase(all_next_nodes.begin() + 1, all_next_nodes.end()); 
            }

            // connect line between current nodes and next nodes
            int next_nodes_num_per_node = all_next_nodes.size() / num_nodes;
            auto next_base_left = layer_base_left_cal(layer);
            auto next_base_top = layer_base_top_cal(all_next_nodes.size());

            // 0: N -> 1;
            // 1: N -> N;
            // 2: N -> M, M==x*N and x is unsigned int
            auto flag = next_nodes_num_per_node > 1 ? 2 : next_nodes_num_per_node;

            auto next_node_top_ = next_base_top; 
            for (int i = 0; i < num_nodes; i++) {
                auto node_left = base_left;
                auto node_top = base_top + i * (node_height + node_gap_vertical);
                // draw blocks connect line between nodes and nodes
                auto draw_connect_block = [=](int next_node_top){                    
                    cv::line(bg_canvas, 
                                cv::Point(node_left + node_width + node_queue_port_w_h, node_top + node_title_h + node_queue_port_padding + node_queue_port_w_h / 2), 
                                cv::Point(node_left + node_width + node_gap_horizontal / 2, node_top + node_title_h + node_queue_port_padding + node_queue_port_w_h / 2),
                                cv::Scalar(156, 156, 156), 1);
                    cv::line(bg_canvas, 
                            cv::Point(node_left + node_width + node_gap_horizontal / 2, node_top + node_title_h + node_queue_port_padding + node_queue_port_w_h / 2), 
                            cv::Point(node_left + node_width + node_gap_horizontal / 2, next_node_top + node_height - node_queue_port_padding - node_queue_port_w_h / 2),
                            cv::Scalar(156, 156, 156), 1);
                    cv::line(bg_canvas, 
                            cv::Point(node_left + node_width + node_gap_horizontal / 2, next_node_top + node_height - node_queue_port_padding - node_queue_port_w_h / 2), 
                            cv::Point(node_left + node_width + node_gap_horizontal - node_queue_port_w_h, next_node_top + node_height - node_queue_port_padding - node_queue_port_w_h / 2),
                            cv::Scalar(156, 156, 156), 1);

                    std::vector<cv::Point> vertexs {cv::Point(node_left + node_width + node_gap_horizontal - node_queue_port_w_h, next_node_top + node_height - node_queue_port_padding - node_queue_port_w_h / 2), 
                                                    cv::Point(node_left + node_width + node_gap_horizontal - node_queue_port_w_h * 2, next_node_top + node_height - node_queue_port_padding - node_queue_port_w_h), 
                                                    cv::Point(node_left + node_width + node_gap_horizontal - node_queue_port_w_h * 2, next_node_top + node_height - node_queue_port_padding)};
                    cv::fillPoly(bg_canvas, std::vector<std::vector<cv::Point>>{vertexs}, cv::Scalar(156, 156, 156));};
                
                if (flag == 0) {
                    next_node_top_ = next_base_top;
                }
                if (flag == 1) {
                    next_node_top_ = next_base_top + i * (node_height + node_gap_vertical);
                }

                if (flag == 0 || flag == 1) {
                    draw_connect_block(next_node_top_);
                }
                else {
                    for (int j = 0; j < next_nodes_num_per_node; j++) {
                        next_node_top_ = next_base_top + (i * next_nodes_num_per_node + j) * (node_height + node_gap_vertical);
                        draw_connect_block(next_node_top_);
                    }
                }
            }

            // render next layer
            render_layer(all_next_nodes, layer + 1);
        }
        // recursion end 
    }
}