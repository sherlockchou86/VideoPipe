

#include "vp_ba_socket_broker_node.h"


namespace vp_nodes {
        
    vp_ba_socket_broker_node::vp_ba_socket_broker_node(std::string node_name,
                                                        std::string des_ip,
                                                        int des_port,
                                                        vp_broke_for broke_for, 
                                                        int broking_cache_warn_threshold, 
                                                        int broking_cache_ignore_threshold):
                                                        vp_msg_broker_node(node_name, broke_for, broking_cache_warn_threshold, broking_cache_ignore_threshold),
                                                        des_ip(des_ip),
                                                        des_port(des_port) {
        // only for vp_frame_target since BA logic ONLY works on vp_frame_target                                                   
        assert(broke_for == vp_broke_for::NORMAL);
        udp_writer = kissnet::udp_socket(kissnet::endpoint(des_ip, des_port));
        VP_INFO(vp_utils::string_format("[%s] [message broker] set des_ip as `%s` and des_port as [%d]", node_name.c_str(), des_ip.c_str(), des_port));
        this->initialized();
    }
    
    vp_ba_socket_broker_node::~vp_ba_socket_broker_node() {
        deinitialized();
        stop_broking();
    }

    void vp_ba_socket_broker_node::format_msg(const std::shared_ptr<vp_objects::vp_frame_meta>& meta, std::string& msg) {
        /* format:
        line 0, <--
        line 1, time
        line 2, channel_index, frame_index
        line 3, ba_type, ba_label
        line 4, ids of involve targets (target_id_1, target_id_2,...)
        line 5, vertexs of involve region (x1,y1,x2,y2,...) 
        line 6, property labels split by '|' and split by ','  between different involve targets
        line 7, record image path
        line 8, record video path
        line 9, -->
        line 10, <--
        line 11, time
        line 12, channel_index, frame_index
        line 13, ba_type, ba_label
        line 14, ids of involve targets (target_id_1, target_id_2,...)
        line 15, vertexs of involve region (x1,y1,x2,y2,...) 
        line 16, property labels split by '|' and split by ','  between different involve targets
        line 17, record image path
        linr 18, record video path
        line 19, -->
        line 20, ...
        */

        std::stringstream msg_stream;
        auto format_basic_info = [&](int channel_index, int frame_index) {
            msg_stream << vp_utils::time_format(NOW) << std::endl;           // line1
            msg_stream << channel_index << "," << frame_index << std::endl;  // line2
        };
        auto format_ba_info = [&](vp_objects::vp_ba_type ba_type,
                                std::string ba_label,
                                const std::vector<int>& involve_target_ids, 
                                const std::vector<vp_objects::vp_point>& involve_region_vertexs) {
            msg_stream << int(ba_type) << "," << ba_label << std::endl;     // line3
            for (int i = 0; i < involve_target_ids.size(); i++) {
                msg_stream << involve_target_ids[i];  // line 4
                if (i != involve_target_ids.size() - 1) {
                    msg_stream << ",";
                }
            }
            msg_stream << std::endl;
            
            for (int i = 0; i < involve_region_vertexs.size(); i++) {
                msg_stream << involve_region_vertexs[i].x << "," << involve_region_vertexs[i].y;  // line 5
                if (i != involve_region_vertexs.size() - 1) {
                    msg_stream << ",";
                }
            }
            msg_stream << std::endl;

            auto targets = meta->get_targets_by_ids(involve_target_ids);
            for (int i = 0; i < targets.size(); i++) {
                auto t = targets[i];
                for (int j = 0; j < t->secondary_labels.size(); j++) {
                    msg_stream << t->secondary_labels[j];  // line 6
                    if (j != t->secondary_labels.size() - 1) {
                        msg_stream << "|";
                    }
                }
                if (i != targets.size() - 1) {
                    msg_stream << ",";
                }
            }
            msg_stream << std::endl;
        };
        auto format_record_info = [&](const std::string& record_image_name, const std::string& record_video_name) {
            msg_stream << record_image_name << std::endl;  // line7
            msg_stream << record_video_name << std::endl;  // line8
        };

        if (broke_for == vp_broke_for::NORMAL) {
            for (int i = 0; i < meta->ba_results.size(); i++) {
                auto& ba = meta->ba_results[i];
                
                // start flag
                msg_stream << "<--" << std::endl;
                // basic info
                format_basic_info(meta->channel_index, meta->frame_index);
                //ba info
                format_ba_info(ba->type, ba->ba_label, ba->involve_target_ids_in_frame, ba->involve_region_in_frame);
                // record info
                format_record_info(ba->record_image_name, ba->record_video_name);
                // end flag
                msg_stream << "-->";
                
                if (i != meta->ba_results.size() - 1) {
                    msg_stream << std::endl;  // not the last one
                }
            }
        }
        
        msg = msg_stream.str();   
    }

    void vp_ba_socket_broker_node::broke_msg(const std::string& msg) {
        // broke msg to socket by udp
        auto bytes_2_send = reinterpret_cast<const std::byte*>(msg.c_str());
        auto bytes_2_send_len = msg.size();
        udp_writer.send(bytes_2_send, bytes_2_send_len);
    }
}