
#pragma once

#include "../vp_node.h" 
/*
* ################################
* why need osd in our pipeline?
* ################################
* there are several reasons why we need osd,
* 1. we need debug, for the outputs of infer, tracker and ba, displaying is more straightforward than printing.
* 2. in some situations like cast screen, it is a functional requirement that we need display what we have done on screen to show what we can do to others who do not know about our product.
* 
* drawing the targets on current frame is the most common operation for osd.
*/
namespace vp_nodes {
    // config for vp_osd_node, define how to draw
    typedef struct vp_osd_node_option
    {
        int aaa = 0;
    } vp_osd_option;
    

    // on screen display(short as osd) node.
    // mainly used to display vp_frame_target on frame.
    class vp_osd_node: public vp_node {
    private:
    protected:
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override;
        virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) override;
    public:
        vp_osd_node(std::string node_name, vp_osd_option options);
        ~vp_osd_node();

        vp_osd_option osd_options;
    };

}