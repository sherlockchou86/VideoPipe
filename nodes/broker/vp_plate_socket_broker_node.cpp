

#include "vp_plate_socket_broker_node.h"


namespace vp_nodes {
        
    vp_plate_socket_broker_node::vp_plate_socket_broker_node(std::string node_name,
                                                        std::string des_ip,
                                                        int des_port,
                                                        std::string plates_dir,
                                                        int min_crop_width,
                                                        int min_crop_height,
                                                        vp_broke_for broke_for, 
                                                        bool only_for_tracked,
                                                        int broking_cache_warn_threshold, 
                                                        int broking_cache_ignore_threshold):
                                                        vp_msg_broker_node(node_name, broke_for, broking_cache_warn_threshold, broking_cache_ignore_threshold),
                                                        des_ip(des_ip),
                                                        des_port(des_port),
                                                        plates_dir(plates_dir),
                                                        min_crop_width(min_crop_width),
                                                        min_crop_height(min_crop_height),
                                                        only_for_tracked(only_for_tracked) {
        // only for vp_frame_target                                                    
        assert(broke_for == vp_broke_for::NORMAL);
        udp_writer = kissnet::udp_socket(kissnet::endpoint(des_ip, des_port));
        VP_INFO(vp_utils::string_format("[%s] [message broker] set des_ip as `%s` and des_port as [%d]", node_name.c_str(), des_ip.c_str(), des_port));
        this->initialized();
    }
    
    vp_plate_socket_broker_node::~vp_plate_socket_broker_node() {
        deinitialized();
        stop_broking();
    }

    void vp_plate_socket_broker_node::format_msg(const std::shared_ptr<vp_objects::vp_frame_meta>& meta, std::string& msg) {
        /* format:
        line 0, <--
        line 1, 1st time
        line 2, 1st channel index, frame index
        line 3, 1st the cropped image's path
        line 4, 1st the whole image's path
        line 5, 1st plate color
        line 6, 1st plate text
        line 7, -->
        line 8, <--
        line 9, 2nd time
        line 10, 2nd channel index, frame index
        line 11, 2nd the cropped image's path
        line 12, 2nd the whole image's path
        line 13, 2nd plate color
        line 14, 2nd plate text
        line 15, -->
        line 16, ...
        */
        auto& broked_ids = all_broked_ids[meta->channel_index];
        auto& broked_texts = all_broked_texts[meta->channel_index];
        // remove 50 elements every 100 ids
        if (broked_ids.size() > 100) {
            broked_ids.erase(broked_ids.begin(), broked_ids.begin() + 50);
        }
        if (broked_texts.size() > 100) {
            broked_texts.erase(broked_texts.begin(), broked_texts.begin() + 50);
        }

        std::stringstream msg_stream;
        auto format_basic_info = [&](int channel_index, int frame_index) {
            msg_stream << vp_utils::time_format(NOW) << std::endl;           // line1
            msg_stream << channel_index << "," << frame_index << std::endl;  // line2
        };
        auto save_cropped_image = [&](cv::Mat& frame, cv::Rect rect, std::string name) {
            auto cropped = frame(rect);
            cv::imwrite(name, cropped);
            msg_stream << name << std::endl;  // line3
        };
        auto save_whole_image = [&](cv::Mat& frame, std::string name) {
            cv::imwrite(name, frame);
            msg_stream << name << std::endl;  // line4
        };
        auto format_plate = [&](const std::string& plate_color, const std::string& plate_text) {
            msg_stream << plate_color << std::endl;  // line5          
            msg_stream << plate_text << std::endl;   // line6
        };
        if (broke_for == vp_broke_for::NORMAL) {
            for (int i = 0; i < meta->targets.size(); i++) {
                auto& t = meta->targets[i];
                // only broke for tracked targets and have enough frames
                if ((only_for_tracked && t->track_id < 0) || (only_for_tracked && t->tracks.size() < min_tracked_frames)) {
                    continue;
                }
                
                // only broke 1 time for specific track id if it has been tracked, or broke many times
                if (t->track_id >= 0 && 
                    std::find(broked_ids.begin(), broked_ids.end(), t->track_id) != broked_ids.end()) {
                    broked_ids.push_back(t->track_id);
                    continue;
                }

                // size filter
                if (t->width < min_crop_width || t->height < min_crop_height) {
                    continue;
                }

                auto color_and_text = vp_utils::string_split(t->primary_label, '_');
                // make sure plate text and color detected
                if(color_and_text.size() != 2) {
                    continue;
                }
                auto& color = color_and_text[0];
                auto& text = color_and_text[1];
                // check for length, 7 or 3 + 6 or 3 + 7
                if (text.length() != 7 && text.length() != 9 && text.length() != 10) {
                    continue;
                }

                // only broke 1 time for specific plate text in small period
                if (std::find(broked_texts.begin(), broked_texts.end(), text) != broked_texts.end()) {
                    broked_texts.push_back(text);
                    continue;
                }
                
                // cache for texts
                broked_texts.push_back(text);
                // cache for ids
                if (t->track_id >= 0) {
                    broked_ids.push_back(t->track_id);
                }

                auto cropped_name = plates_dir + "/" + std::to_string(t->channel_index) + "_" + std::to_string(t->frame_index) + "_" + color + "_" + text  + "_cropped.jpg";
                auto whole_name = plates_dir + "/" + std::to_string(t->channel_index) + "_" + std::to_string(t->frame_index) + "_" + color + "_" + text  + "_whole.jpg";
                // start flag
                msg_stream << "<--" << std::endl;
                // basic info
                format_basic_info(meta->channel_index, meta->frame_index);
                // save small cropped image
                save_cropped_image(meta->frame, cv::Rect(t->x, t->y, t->width, t->height), cropped_name);
                // save whole image
                save_whole_image(meta->frame, whole_name);
                // format color and text
                format_plate(color, text);
                // end flag
                msg_stream << "-->";
                
                if (i != meta->targets.size() - 1) {
                    msg_stream << std::endl;  // not the last one
                }
            }
        }
        
        msg = msg_stream.str();   
    }

    void vp_plate_socket_broker_node::broke_msg(const std::string& msg) {
        // broke msg to socket by udp
        auto bytes_2_send = reinterpret_cast<const std::byte*>(msg.c_str());
        auto bytes_2_send_len = msg.size();
        udp_writer.send(bytes_2_send, bytes_2_send_len);
    }
}