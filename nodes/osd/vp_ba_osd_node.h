#pragma once

#include <opencv2/freetype.hpp>
#include "../vp_node.h"
#include "../../objects/shapes/vp_point.h"

namespace vp_nodes {
    // osd node for behaviour analysis 
    class vp_ba_osd_node: public vp_node
    {
    private:
        // support chinese font
        cv::Ptr<cv::freetype::FreeType2> ft2;
    protected:
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override;
    public:
        vp_ba_osd_node(std::string node_name,  std::string font = "");
        ~vp_ba_osd_node();
    };
}