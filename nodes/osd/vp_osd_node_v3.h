#pragma once

#include <opencv2/freetype.hpp>

#include "../vp_node.h"


namespace vp_nodes {
    // on screen display(short as osd) node.
    // another version for vp_frame_target display, display mask area for image segmentation.
    class vp_osd_node_v3: public vp_node
    {
    private:
        // support chinese font
        cv::Ptr<cv::freetype::FreeType2> ft2;
        float mask_threshold = 0.3;
    protected:
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override;
    public:
        vp_osd_node_v3(std::string node_name, std::string font = "");
        ~vp_osd_node_v3();
    };
}