
#include "vp_app_des_node.h"

namespace vp_nodes {
        
    vp_app_des_node::vp_app_des_node(std::string node_name, 
                                        int channel_index): 
                                        vp_des_node(node_name, channel_index) {
        this->initialized();
    }
    
    vp_app_des_node::~vp_app_des_node() {
        deinitialized();
    }
    
    // re-implementation, return nullptr.
    std::shared_ptr<vp_objects::vp_meta> 
        vp_app_des_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
            VP_DEBUG(vp_utils::string_format("[%s] received frame meta, channel_index=>%d, frame_index=>%d", node_name.c_str(), meta->channel_index, meta->frame_index));

            invoke_app_des_result_hooker(meta);

            // for general works defined in base class
            return vp_des_node::handle_frame_meta(meta);
    }

    // re-implementation, return nullptr.
    std::shared_ptr<vp_objects::vp_meta> 
        vp_app_des_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
            invoke_app_des_result_hooker(meta);
            return vp_des_node::handle_control_meta(meta);
    }

    void vp_app_des_node::set_app_des_result_hooker(vp_app_des_result_hooker app_des_result_hooker) {
        this->app_des_result_hooker = app_des_result_hooker;
    }

    void vp_app_des_node::invoke_app_des_result_hooker(std::shared_ptr<vp_objects::vp_meta> meta) {
        if (app_des_result_hooker) {
            app_des_result_hooker(node_name, meta);
        }
    }
}