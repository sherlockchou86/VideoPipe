#pragma once

#include "vp_des_node.h"
#include "../objects/vp_frame_meta.h"
#include "../objects/vp_control_meta.h"

#include <iostream>
#include <memory>

namespace vp_nodes {
    class vp_file_des_node: public vp_des_node {
    private:
        /* data */
    protected:
        // re-implementation, return nullptr.
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override; 
        // re-implementation, return nullptr.
        virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) override;
    public:
        vp_file_des_node(std::string node_name, 
                        int channel_index, 
                        std::string file_path);
        ~vp_file_des_node();

        std::string file_path;
    };

}