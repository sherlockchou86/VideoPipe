
#include <opencv2/imgproc.hpp>
#include "vp_expr_osd_node.h"
#include "../../utils/vp_utils.h"


namespace vp_nodes {
    
    vp_expr_osd_node::vp_expr_osd_node(std::string node_name, std::string font):vp_node(node_name) {
        assert(font != "");
        ft2 = cv::freetype::createFreeType2();
        ft2->loadFontData(font, 0);    
        this->initialized();  
    }
    
    vp_expr_osd_node::~vp_expr_osd_node() {
        deinitialized();
    }

    std::shared_ptr<vp_objects::vp_meta> vp_expr_osd_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        // operations on osd_frame
        if (meta->osd_frame.empty()) {
            meta->osd_frame = meta->frame.clone();
        }
        auto& canvas = meta->osd_frame;

        for (auto& i: meta->text_targets) {
            cv::Point rook_points[4];
            for (int m = 0; m < i->region_vertexes.size(); m++) {
                rook_points[m] =
                    cv::Point(i->region_vertexes[m].first, i->region_vertexes[m].second);
            }

            const cv::Point *ppt[1] = {rook_points};
            int npt[] = {4};

            if (i->flags.find("yes") != std::string::npos) {
                cv::polylines(canvas, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, cv::LINE_AA, 0);   // green
                ft2->putText(canvas, "√", rook_points[1], 30, CV_RGB(0, 255, 0), cv::FILLED, cv::LINE_AA, true);
            }
            else if (i->flags.find("no") != std::string::npos) {
                auto right_value = vp_utils::string_split(i->flags, '_')[1];
                cv::polylines(canvas, ppt, npt, 1, 1, CV_RGB(255, 0, 0), 2, cv::LINE_AA, 0);   // red
                ft2->putText(canvas, "×(" + right_value + ")", rook_points[1], 30, CV_RGB(255, 0, 0), cv::FILLED, cv::LINE_AA, true);
            }
            else if (i->flags == "invalid") {
                cv::polylines(canvas, ppt, npt, 1, 1, CV_RGB(255, 165, 0), 2, cv::LINE_AA, 0); // orange
                ft2->putText(canvas, "invalid", rook_points[1], 30, CV_RGB(255, 165, 0), cv::FILLED, cv::LINE_AA, true);
            }
            else {
                // to-do
            }
        }
        return meta;
    }
}