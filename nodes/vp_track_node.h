
#pragma once

#include "vp_node.h"

namespace vp_nodes {
    class vp_track_node: public vp_node
    {
    private:
        /* data */

    protected:
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override;
        virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) override;
    public:
        vp_track_node(std::string node_name);
        ~vp_track_node();
    };
}