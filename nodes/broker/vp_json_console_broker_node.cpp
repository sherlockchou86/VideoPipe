
#include "vp_json_console_broker_node.h"

namespace vp_nodes {
        
    vp_json_console_broker_node::vp_json_console_broker_node(std::string node_name, 
                                                            vp_broke_for broke_for, 
                                                            int broking_cache_warn_threshold, 
                                                            int broking_cache_ignore_threshold):
                                                            vp_msg_broker_node(node_name, broke_for, broking_cache_warn_threshold, broking_cache_ignore_threshold) {
        this->initialized();
    }
    
    vp_json_console_broker_node::~vp_json_console_broker_node() {
        deinitialized();
        stop_broking();
    }
    
    void vp_json_console_broker_node::format_msg(const std::shared_ptr<vp_objects::vp_frame_meta>& meta, std::string& msg) {
        // serialize objects to json by cereal
        std::stringstream msg_stream;
        {
            cereal::JSONOutputArchive json_archive(msg_stream);
            
            // global values
            json_archive(cereal::make_nvp("channel_index", meta->channel_index),
            cereal::make_nvp("frame_index", meta->frame_index),
            cereal::make_nvp("width", meta->frame.cols),
            cereal::make_nvp("height", meta->frame.rows),
            cereal::make_nvp("fps", meta->fps),
            cereal::make_nvp("broke_for", broke_fors.at(broke_for)));

            // serialize values according to broke_for
            if (broke_for == vp_broke_for::NORMAL) {
                json_archive(cereal::make_nvp("target_size", meta->targets.size()), 
                            cereal::make_nvp("targets", meta->targets));
            }
            else if (broke_for ==  vp_broke_for::FACE) {
                json_archive(cereal::make_nvp("face_target_size", meta->face_targets.size()),
                            cereal::make_nvp("face_targets", meta->face_targets));
            }
            else if (broke_for == vp_broke_for::TEXT) {
                json_archive(cereal::make_nvp("text_target_size", meta->text_targets.size()),
                            cereal::make_nvp("text_targets", meta->text_targets));
            }
            else {
                throw "invalid broke_for!";
            }
        } // flush

        msg = msg_stream.str();
    }

    void vp_json_console_broker_node::broke_msg(const std::string& msg) {
        // broke msg to console by std::cout
        std::cout << msg << std::endl;
    }
}