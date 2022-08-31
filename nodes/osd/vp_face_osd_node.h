#pragma once

#include "../vp_node.h"

namespace vp_nodes {
    // on screen display(short as osd) node.
    // mainly used to display vp_frame_face_target on frame.
    class vp_face_osd_node: public vp_node
    {
    private:
        /* data */
    protected:
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override;
        virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) override;
    public:
        vp_face_osd_node(std::string node_name);
        ~vp_face_osd_node();
    };

}