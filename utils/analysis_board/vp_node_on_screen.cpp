
#include "vp_node_on_screen.h"


namespace vp_utils {
        
    vp_node_on_screen::vp_node_on_screen(std::shared_ptr<vp_nodes::vp_node> original_node, 
                                        vp_objects::vp_rect node_rect, 
                                        int layer):
                                        original_node(original_node),
                                        node_rect(node_rect),
                                        layer(layer) {
        assert(original_node != nullptr);
        // register meta hookers for all nodes
        original_node->set_meta_arriving_hooker([this](std::string node_name, int queue_size, std::shared_ptr<vp_objects::vp_meta> meta) {
                this->meta_arriving_hooker_storage.meta = meta;
                this->meta_arriving_hooker_storage.queue_size = queue_size;
                this->meta_arriving_hooker_storage.called_count_since_epoch_start++;
            });
        original_node->set_meta_handling_hooker([this](std::string node_name, int queue_size, std::shared_ptr<vp_objects::vp_meta> meta) {
                this->meta_handling_hooker_storage.meta = meta;
                this->meta_handling_hooker_storage.queue_size = queue_size;
                this->meta_handling_hooker_storage.called_count_since_epoch_start++;
            });
        original_node->set_meta_handled_hooker([this](std::string node_name, int queue_size, std::shared_ptr<vp_objects::vp_meta> meta) {
                this->meta_handled_hooker_storage.meta = meta;
                this->meta_handled_hooker_storage.queue_size = queue_size;
                this->meta_handled_hooker_storage.called_count_since_epoch_start++;
            });
        original_node->set_meta_leaving_hooker([this](std::string node_name, int queue_size, std::shared_ptr<vp_objects::vp_meta> meta) {
                this->meta_leaving_hooker_storage.meta = meta;
                this->meta_leaving_hooker_storage.queue_size = queue_size;
                this->meta_leaving_hooker_storage.called_count_since_epoch_start++;
            });
        
        // register stream info hooker if it is a src node
        if (original_node->node_type() == vp_nodes::vp_node_type::SRC) {
            auto src_node = std::dynamic_pointer_cast<vp_nodes::vp_src_node>(original_node);
            src_node->set_stream_info_hooker([this](std::string node_name, vp_nodes::vp_stream_info stream_info) {
                this->stream_info_hooker_storage = stream_info;
            });
        }
        if (original_node->node_type() == vp_nodes::vp_node_type::DES) {
            auto des_node = std::dynamic_pointer_cast<vp_nodes::vp_des_node>(original_node);
            des_node->set_stream_status_hooker([this](std::string node_name, vp_nodes::vp_stream_status stream_status){
                this->stream_status_hooker_storage = stream_status;
            });
        }
        
    }
    
    vp_node_on_screen::~vp_node_on_screen() {
        // unregister meta hookers for all nodes
        original_node->set_meta_arriving_hooker({});
        original_node->set_meta_handling_hooker({});
        original_node->set_meta_handled_hooker({});
        original_node->set_meta_leaving_hooker({});
        
        // unregister stream info hooker if it is a src node
        if (original_node->node_type() == vp_nodes::vp_node_type::SRC) {
            auto src_node = std::dynamic_pointer_cast<vp_nodes::vp_src_node>(original_node);
            src_node->set_stream_info_hooker({});
        }
        if (original_node->node_type() == vp_nodes::vp_node_type::DES) {
            auto des_node = std::dynamic_pointer_cast<vp_nodes::vp_des_node>(original_node);
            des_node->set_stream_status_hooker({});
        }
    }
    
    void vp_node_on_screen::render_static_parts(cv::Mat & canvas) {
        auto node_left = node_rect.x;
        auto node_top = node_rect.y;
        auto node_width = node_rect.width;
        auto node_height = node_rect.height;
        
        cv::rectangle(canvas, cv::Rect(node_left, node_top, node_width, node_height), cv::Scalar(0, 0, 0), 1);
        // node_name
        vp_utils::put_text_at_center_of_rect(canvas, original_node->node_name, cv::Rect(node_left, node_top + 1, node_width, node_title_h - 2), false, font_face);
        cv::line(canvas, 
                cv::Point(node_left, node_top + node_title_h), 
                cv::Point(node_left + node_width - 1, node_top + node_title_h), 
                cv::Scalar(0, 0, 0), 1);

        // draw in_queue for non-src nodes
        if (original_node->node_type() != vp_nodes::vp_node_type::SRC) {
            // connect line between in_queue and out_queue
            if (original_node->node_type() == vp_nodes::vp_node_type::MID) {
                cv::line(canvas, 
                        cv::Point(node_left + node_queue_width + node_queue_port_w_h, node_top + node_title_h + node_queue_port_padding + node_queue_port_w_h / 2), 
                        cv::Point(node_left + node_width / 2, node_top + node_title_h + node_queue_port_padding + node_queue_port_w_h / 2),
                        cv::Scalar(156, 156, 156), 1);
                cv::line(canvas, 
                        cv::Point(node_left + node_width / 2, node_top + node_title_h + node_queue_port_padding + node_queue_port_w_h / 2), 
                        cv::Point(node_left + node_width / 2, node_top + node_height - node_queue_port_padding - node_queue_port_w_h / 2),
                        cv::Scalar(156, 156, 156), 1);
                cv::line(canvas, 
                        cv::Point(node_left + node_width / 2, node_top + node_height - node_queue_port_padding - node_queue_port_w_h / 2), 
                        cv::Point(node_left + node_width - node_queue_width - node_queue_port_w_h, node_top + node_height - node_queue_port_padding - node_queue_port_w_h / 2),
                        cv::Scalar(156, 156, 156), 1);
                std::vector<cv::Point> vertexs {cv::Point(node_left + node_width - node_queue_width - node_queue_port_w_h, node_top + node_height - node_queue_port_padding - node_queue_port_w_h / 2), 
                                                cv::Point(node_left + node_width - node_queue_width - node_queue_port_w_h * 2, node_top + node_height - node_queue_port_padding - node_queue_port_w_h), 
                                                cv::Point(node_left + node_width - node_queue_width - node_queue_port_w_h * 2, node_top + node_height - node_queue_port_padding)};
                cv::fillPoly(canvas, std::vector<std::vector<cv::Point>>{vertexs}, cv::Scalar(156, 156, 156));
            }
            else {
                // DES
                cv::line(canvas, 
                        cv::Point(node_left + node_queue_width + node_queue_port_w_h, node_top + node_title_h + node_queue_port_padding + node_queue_port_w_h / 2), 
                        cv::Point(node_left + node_width / 2, node_top + node_title_h + node_queue_port_padding + node_queue_port_w_h / 2),
                        cv::Scalar(156, 156, 156), 1);
                std::vector<cv::Point> vertexs {cv::Point(node_left + node_width / 2, node_top + node_title_h + node_queue_port_padding), 
                                                cv::Point(node_left + node_width / 2, node_top + node_title_h + node_queue_port_padding + node_queue_port_w_h), 
                                                cv::Point(node_left + node_width / 2 + node_queue_port_w_h, node_top+ node_title_h + node_queue_port_padding + node_queue_port_w_h / 2)};
                cv::fillPoly(canvas, std::vector<std::vector<cv::Point>>{vertexs}, cv::Scalar(156, 156, 156));
            }

            cv::line(canvas, 
                    cv::Point(node_left + node_queue_width, node_top + node_title_h), 
                    cv::Point(node_left + node_queue_width, node_top + node_height - 1), cv::Scalar(0, 0, 0), 1);

            // in port
            cv::rectangle(canvas, 
                            cv::Rect(node_left - node_queue_port_w_h + 1, node_top + node_height - node_queue_port_padding - node_queue_port_w_h, node_queue_port_w_h, node_queue_port_w_h), 
                            cv::Scalar(156, 156, 156), 1);

            // out port
            cv::rectangle(canvas, 
                            cv::Rect(node_left + node_queue_width, node_top + node_title_h + node_queue_port_padding, node_queue_port_w_h, node_queue_port_w_h), 
                            cv::Scalar(156, 156, 156), 1);
        }
        // draw out_queue for non-des nodes
        if (original_node->node_type() != vp_nodes::vp_node_type::DES) {
            // connect line between in_queue and out_queue 
            if (original_node->node_type() == vp_nodes::vp_node_type::SRC) {
                cv::line(canvas, 
                        cv::Point(node_left + node_width / 2, node_top + node_height - node_queue_port_padding - node_queue_port_w_h / 2), 
                        cv::Point(node_left + node_width - node_queue_width - node_queue_port_w_h, node_top + node_height - node_queue_port_padding - node_queue_port_w_h / 2), 
                        cv::Scalar(156, 156, 156), 1);
                
                std::vector<cv::Point> vertexs {cv::Point(node_left + node_width - node_queue_width - node_queue_port_w_h, node_top + node_height - node_queue_port_padding - node_queue_port_w_h / 2), 
                                                cv::Point(node_left + node_width - node_queue_width - node_queue_port_w_h * 2, node_top + node_height - node_queue_port_padding - node_queue_port_w_h), 
                                                cv::Point(node_left + node_width - node_queue_width - node_queue_port_w_h * 2, node_top + node_height - node_queue_port_padding)};
                cv::fillPoly(canvas, std::vector<std::vector<cv::Point>>{vertexs}, cv::Scalar(156, 156, 156));
            }
            
            cv::line(canvas, 
                    cv::Point(node_left + node_width - node_queue_width, node_top + node_title_h), 
                    cv::Point(node_left + node_width - node_queue_width, node_top + node_height - 1), 
                    cv::Scalar(0, 0, 0), 1);
            
            // in port
            cv::rectangle(canvas, 
                            cv::Rect(node_left + node_width - node_queue_width - node_queue_port_w_h + 1, node_top + node_height - node_queue_port_padding - node_queue_port_w_h, node_queue_port_w_h, node_queue_port_w_h), 
                            cv::Scalar(156, 156, 156), 1);
            // out port
            cv::rectangle(canvas,
                            cv::Rect(node_left + node_width -1 , node_top + node_title_h + node_queue_port_padding, node_queue_port_w_h, node_queue_port_w_h), 
                            cv::Scalar(156, 156, 156), 1);
        }

        // draw blocks connect line between nodes and nodes
        auto draw_connect_block = [=](int next_node_top){                    
            cv::line(canvas, 
                        cv::Point(node_left + node_width + node_queue_port_w_h, node_top + node_title_h + node_queue_port_padding + node_queue_port_w_h / 2), 
                        cv::Point(node_left + node_width + node_gap_horizontal / 2, node_top + node_title_h + node_queue_port_padding + node_queue_port_w_h / 2),
                        cv::Scalar(156, 156, 156), 1);
            cv::line(canvas, 
                    cv::Point(node_left + node_width + node_gap_horizontal / 2, node_top + node_title_h + node_queue_port_padding + node_queue_port_w_h / 2), 
                    cv::Point(node_left + node_width + node_gap_horizontal / 2, next_node_top + node_height - node_queue_port_padding - node_queue_port_w_h / 2),
                    cv::Scalar(156, 156, 156), 1);
            cv::line(canvas, 
                    cv::Point(node_left + node_width + node_gap_horizontal / 2, next_node_top + node_height - node_queue_port_padding - node_queue_port_w_h / 2), 
                    cv::Point(node_left + node_width + node_gap_horizontal - node_queue_port_w_h, next_node_top + node_height - node_queue_port_padding - node_queue_port_w_h / 2),
                    cv::Scalar(156, 156, 156), 1);

            std::vector<cv::Point> vertexs {cv::Point(node_left + node_width + node_gap_horizontal - node_queue_port_w_h, next_node_top + node_height - node_queue_port_padding - node_queue_port_w_h / 2), 
                                            cv::Point(node_left + node_width + node_gap_horizontal - node_queue_port_w_h * 2, next_node_top + node_height - node_queue_port_padding - node_queue_port_w_h), 
                                            cv::Point(node_left + node_width + node_gap_horizontal - node_queue_port_w_h * 2, next_node_top + node_height - node_queue_port_padding)};
            cv::fillPoly(canvas, std::vector<std::vector<cv::Point>>{vertexs}, cv::Scalar(156, 156, 156));};
        
        auto next_nodes_num = next_nodes_on_screen.size();
        for (int j = 0; j < next_nodes_num; j++) {
            draw_connect_block(next_nodes_on_screen[j]->node_rect.y);
        }     
    }

    void vp_node_on_screen::render_dynamic_parts(cv::Mat & canvas) {
        /*
        * draw data from hookers' callbacks
        */
        auto fps_func = [&](vp_meta_hooker_storage& storage, cv::Rect rect) {
            auto called_count = storage.called_count_since_epoch_start;
            auto epoch_start = storage.time_epoch_start;
            auto delta_sec = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - epoch_start);
            if (delta_sec.count() > fps_timeout * 1000 || (delta_sec.count() > fps_epoch && called_count > 0)) {
                //int fps = round(called_count * 1000.0 / delta_sec.count());
                auto fps = vp_utils::round_any(called_count * 1000.0 / delta_sec.count(), 1);
                storage.called_count_since_epoch_start = 0;
                storage.time_epoch_start = std::chrono::system_clock::now();
                storage.pre_fps = fps;  // cache for next show
                vp_utils::put_text_at_center_of_rect(canvas, 
                    fps, 
                    rect, true,
                    font_face, 1, cv::Scalar(26, 26, 139));
            }
            else {
                // use previous fps
                vp_utils::put_text_at_center_of_rect(canvas, 
                    storage.pre_fps, 
                    rect, true,
                    font_face, 1, cv::Scalar(26, 26, 139));
            }
        };

        // non-src nodes
        if (original_node->node_type() != vp_nodes::vp_node_type::SRC) { 
            // size of in queue
            vp_utils::put_text_at_center_of_rect(canvas, 
                                    std::to_string(meta_handling_hooker_storage.queue_size), 
                                    cv::Rect(node_rect.x + 3, 
                                    node_rect.y + node_title_h / 2 + (node_rect.height - node_title_h) / 2, node_queue_width  - 8, node_title_h - 10), true, font_face, 1, cv::Scalar(255, 0, 255));
            // fps at 1st port
            fps_func(meta_arriving_hooker_storage, cv::Rect(node_rect.x - node_queue_width / 2, 
                        node_rect.y + node_rect.height - node_queue_port_padding * 3 - node_queue_port_w_h * 3 / 2, 
                        node_queue_width, 
                        node_queue_port_padding + node_queue_port_w_h));
            // fps at 2nd port
            fps_func(meta_handling_hooker_storage, cv::Rect(node_rect.x + node_queue_width - node_queue_width / 2, 
                        node_rect.y + node_title_h + node_queue_port_padding * 3 / 2 + node_queue_port_w_h, 
                        node_queue_width, 
                        node_queue_port_padding + node_queue_port_w_h));
        }   
        
        // non-des nodes
        if (original_node->node_type() != vp_nodes::vp_node_type::DES) { 
            // size of out queue
            vp_utils::put_text_at_center_of_rect(canvas, 
                                    std::to_string(meta_leaving_hooker_storage.queue_size), 
                                    cv::Rect(node_rect.x + node_rect.width - node_queue_width + 3, 
                                    node_rect.y + node_title_h / 2 + (node_rect.height - node_title_h) / 2, node_queue_width - 8, node_title_h - 10), true, font_face, 1, cv::Scalar(255, 0, 255));
            // fps at 3rd port
            fps_func(meta_handled_hooker_storage, cv::Rect(node_rect.x + node_rect.width - node_queue_width - node_queue_width / 2, 
                        node_rect.y + node_rect.height - node_queue_port_padding * 3 - node_queue_port_w_h * 3 / 2, 
                        node_queue_width, 
                        node_queue_port_padding + node_queue_port_w_h));
            // fps at 4th port
            fps_func(meta_leaving_hooker_storage, cv::Rect(node_rect.x + node_rect.width - node_queue_width / 2, 
                        node_rect.y + node_title_h + node_queue_port_padding * 3 / 2 + node_queue_port_w_h, 
                        node_queue_width, 
                        node_queue_port_padding  + node_queue_port_w_h));
        }  

        auto node_left = node_rect.x;
        auto node_top = node_rect.y;
        // stream info at src nodes
        if (original_node->node_type() == vp_nodes::vp_node_type::SRC) {
            vp_utils::put_text_at_center_of_rect(canvas, stream_info_hooker_storage.uri, 
                                                cv::Rect(node_left - node_rect.width * 3 / 4, node_top + node_title_h + node_queue_port_padding, node_rect.width * 4 / 3, node_title_h * 2 / 3), 
                                                true, font_face, 1, cv::Scalar(), cv::Scalar(), cv::Scalar(255, 255, 255));
            vp_utils::put_text_at_center_of_rect(canvas, "original_width: " + std::to_string(stream_info_hooker_storage.original_width),
                                                cv::Rect(node_left - node_rect.width * 3 / 4, node_top + node_title_h * 5 / 3 + node_queue_port_padding * 2, node_rect.width * 4 / 3, node_title_h * 2 / 3),
                                                true, font_face, 1, cv::Scalar(), cv::Scalar(), cv::Scalar(255, 255, 255));
            vp_utils::put_text_at_center_of_rect(canvas, "original_height: " + std::to_string(stream_info_hooker_storage.original_height),
                                                cv::Rect(node_left - node_rect.width * 3 / 4, node_top + node_title_h * 7 / 3 + node_queue_port_padding * 3, node_rect.width * 4 / 3, node_title_h * 2 / 3),
                                                true, font_face, 1, cv::Scalar(), cv::Scalar(), cv::Scalar(255, 255, 255));      
            vp_utils::put_text_at_center_of_rect(canvas, "original_fps: " + std::to_string(stream_info_hooker_storage.original_fps),
                                                cv::Rect(node_left - node_rect.width * 3 / 4, node_top + node_title_h * 9 / 3 + node_queue_port_padding * 4, node_rect.width * 4 / 3, node_title_h * 2 / 3),
                                                true, font_face, 1, cv::Scalar(), cv::Scalar(), cv::Scalar(255, 255, 255));   
        } 
        // stream status at des nodes
        if (original_node->node_type() == vp_nodes::vp_node_type::DES) {
            vp_utils::put_text_at_center_of_rect(canvas, stream_status_hooker_storage.direction,
                                                cv::Rect(node_left + node_rect.width / 2 - 10, node_top + node_title_h * 5 / 3 + node_queue_port_padding * 2, node_rect.width * 4 / 3 + 10, node_title_h * 2 / 3),
                                                true,font_face,1,cv::Scalar(),cv::Scalar(),cv::Scalar(255, 255, 255));    
            vp_utils::put_text_at_center_of_rect(canvas, "output_fps: " + vp_utils::round_any(stream_status_hooker_storage.fps, 2),
                                                cv::Rect(node_left + node_rect.width / 2 - 10, node_top + node_title_h * 7 / 3 + node_queue_port_padding * 3, node_rect.width * 4 / 3 + 10, node_title_h * 2 / 3),
                                                true,font_face,1,cv::Scalar(),cv::Scalar(),cv::Scalar(255, 255, 255));   
            vp_utils::put_text_at_center_of_rect(canvas, "latency: " + std::to_string(stream_status_hooker_storage.latency) + "ms",
                                                cv::Rect(node_left + node_rect.width / 2 - 10, node_top + node_title_h * 9 / 3 + node_queue_port_padding * 4, node_rect.width * 4 / 3 + 10, node_title_h * 2 / 3),
                                                true,font_face,1,cv::Scalar(),cv::Scalar(),cv::Scalar(255, 255, 255));
        }
    }

    std::shared_ptr<vp_nodes::vp_node>& vp_node_on_screen::get_orginal_node() {
        return original_node;
    }

    std::vector<std::shared_ptr<vp_node_on_screen>>& vp_node_on_screen::get_next_nodes_on_screen() {
        return next_nodes_on_screen;
    }
}