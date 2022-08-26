
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "vp_file_des_node.h"

namespace vp_nodes {
        
    vp_file_des_node::vp_file_des_node(std::string node_name, 
                                        int channel_index, 
                                        std::string file_path): 
                                        vp_des_node(node_name, channel_index), 
                                        file_path(file_path) {
        this->initialized();
    }
    
    vp_file_des_node::~vp_file_des_node() {

    }
    
    // re-implementation, return nullptr.
    std::shared_ptr<vp_objects::vp_meta> 
        vp_file_des_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
            return nullptr;
    }

    // re-implementation, return nullptr.
    std::shared_ptr<vp_objects::vp_meta> 
        vp_file_des_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
            return nullptr;
    }
}