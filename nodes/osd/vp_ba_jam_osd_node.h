#pragma once

#include <map>
#include <opencv2/freetype.hpp>
#include "../vp_node.h"
#include "../../objects/shapes/vp_point.h"
#include "../../objects/shapes/vp_line.h"

namespace vp_nodes {
    // osd node for behaviour analysis of stop
    class vp_ba_jam_osd_node: public vp_node
    {
    private:
        // support chinese font
        cv::Ptr<cv::freetype::FreeType2> ft2;

        // support multi channels
        std::map<int, std::vector<vp_objects::vp_point>> all_jam_regions;   // channel -> jam region
        std::map<int, bool> all_jam_results;  // channel -> jam status
        std::map<int, std::vector<int>> all_involve_ids;  // channel -> target ids when enter jam status
    protected:
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override;
    public:
        vp_ba_jam_osd_node(std::string node_name,  std::string font = "");
        ~vp_ba_jam_osd_node();
    };
}