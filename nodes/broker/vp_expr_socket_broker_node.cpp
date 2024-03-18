

#include "vp_expr_socket_broker_node.h"


namespace vp_nodes {
        
    vp_expr_socket_broker_node::vp_expr_socket_broker_node(std::string node_name,
                                                        std::string des_ip,
                                                        int des_port,
                                                        std::string screenshot_dir,
                                                        vp_broke_for broke_for, 
                                                        int broking_cache_warn_threshold, 
                                                        int broking_cache_ignore_threshold):
                                                        vp_msg_broker_node(node_name, broke_for, broking_cache_warn_threshold, broking_cache_ignore_threshold),
                                                        des_ip(des_ip),
                                                        des_port(des_port),
                                                        screenshot_dir(screenshot_dir) {
        // only for vp_frame_text_target                                                 
        assert(broke_for == vp_broke_for::TEXT);
        udp_writer = kissnet::udp_socket(kissnet::endpoint(des_ip, des_port));
        VP_INFO(vp_utils::string_format("[%s] [message broker] set des_ip as `%s` and des_port as [%d]", node_name.c_str(), des_ip.c_str(), des_port));
        this->initialized();
    }
    
    vp_expr_socket_broker_node::~vp_expr_socket_broker_node() {
        deinitialized();
        stop_broking();
    }

    void vp_expr_socket_broker_node::format_msg(const std::shared_ptr<vp_objects::vp_frame_meta>& meta, std::string& msg) {
        /* format:
        line 0, <--
        line 1, time
        line 2, channel_index, frame_index
        line 3, total num, num of yes, num of no, number of invalid
        line 4, 1st expression string, 2nd expression string, 3rd expression string, ...
        line 5, screenshot path
        line 6, -->
        */

        std::stringstream msg_stream;
        auto format_basic_info = [&](int channel_index, int frame_index) {
            msg_stream << vp_utils::time_format(NOW) << std::endl;           // line1
            msg_stream << channel_index << "," << frame_index << std::endl;  // line2
        };
        auto format_expr_info = [&](const std::vector<std::shared_ptr<vp_objects::vp_frame_text_target>>& expr_targets) {
            auto num_yes = 0, num_no = 0, num_invalid = 0;
            auto expr_strs = std::string();
            for (int i = 0; i < expr_targets.size(); i++) {
                expr_strs += expr_targets[i]->text;
                if (i != expr_targets.size() - 1) {
                    expr_strs += ",";
                }

                if (expr_targets[i]->flags.find("yes") != std::string::npos) {
                    num_yes++;
                }
                if (expr_targets[i]->flags.find("no") != std::string::npos) {
                    num_no++;
                }
                if (expr_targets[i]->flags.find("invalid") != std::string::npos) {
                    num_invalid++;
                }
            }
            msg_stream << expr_targets.size() << "," << num_yes << "," << num_no << "," << num_invalid << std::endl;  // line3
            msg_stream << expr_strs << std::endl;   // line4
        };
        auto format_screenshot = [&](cv::Mat& screenshot, const std::string& name) {
            cv::imwrite(name, screenshot);
            msg_stream << name << std::endl;  // line5
        };
        
        // at most 1 record for each frame
        if (broke_for == vp_broke_for::TEXT) {
            // start flag
            msg_stream << "<--" << std::endl;
            format_basic_info(meta->channel_index, meta->frame_index);
            format_expr_info(meta->text_targets);
            auto screenshot_name = screenshot_dir + "/" + std::to_string(meta->channel_index) + "_" + std::to_string(meta->frame_index) + ".jpg";
            format_screenshot(meta->osd_frame.empty() ? meta->frame : meta->osd_frame, screenshot_name);
            // end flag
            msg_stream << "-->" << std::endl;
        }
        
        msg = msg_stream.str();   
    }

    void vp_expr_socket_broker_node::broke_msg(const std::string& msg) {
        // broke msg to socket by udp
        auto bytes_2_send = reinterpret_cast<const std::byte*>(msg.c_str());
        auto bytes_2_send_len = msg.size();
        udp_writer.send(bytes_2_send, bytes_2_send_len);
    }
}