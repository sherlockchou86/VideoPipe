

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include "vp_des_node.h"

namespace vp_nodes {
    // screen des node, display video on local window.
    class vp_screen_des_node: public vp_des_node
    {
    private:
        /* data */
        std::string gst_template = "appsrc ! videoconvert ! videoscale ! textoverlay text=%s halignment=left valignment=top font-desc='Sans,16' shaded-background=true ! timeoverlay halignment=right valignment=top font-desc='Sans,16' shaded-background=true ! queue ! fpsdisplaysink video-sink=ximagesink sync=false";
        cv::VideoWriter screen_writer;
    protected:
        // re-implementation, return nullptr.
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override; 
        // re-implementation, return nullptr.
        virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) override;
    public:
        vp_screen_des_node(std::string node_name, 
                            int channel_index, 
                            bool osd = true,
                            vp_objects::vp_size display_w_h = {});
        ~vp_screen_des_node();

        // for osd frame
        bool osd;
        // display size
        vp_objects::vp_size display_w_h;
    };
}