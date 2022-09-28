#pragma once

#include <opencv2/freetype.hpp>

#include "../vp_node.h"

namespace vp_nodes {
    // on screen display(short as osd) node.
    // another version for vp_frame_target display, display vp_sub_target at the bottom of screen.
    class vp_osd_node_v2: public vp_node
    {
    private:
        // support chinese font
        cv::Ptr<cv::freetype::FreeType2> ft2;
                
        // leave a gap at the bottom of osd frame
        int gap_height = 256;
        int padding = 10;
    protected:
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override;
    public:
        vp_osd_node_v2(std::string node_name, std::string font = "");
        ~vp_osd_node_v2();
    };

}