
#include "vp_text_osd_node.h"

namespace vp_nodes {
        
    vp_text_osd_node::vp_text_osd_node(std::string node_name, std::string font): vp_node(node_name) {
        ft2 = cv::freetype::createFreeType2();
        ft2->loadFontData(font, 0);
        this->initialized();
    }
    
    vp_text_osd_node::~vp_text_osd_node() {
        deinitialized();
    }

    std::shared_ptr<vp_objects::vp_meta> vp_text_osd_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        // operations on osd_frame
        if (meta->osd_frame.empty()) {
            // double times higher than frame 
            meta->osd_frame = cv::Mat(meta->frame.rows * 2, meta->frame.cols, meta->frame.type());
        }

        auto canvas1 = meta->frame.clone();
        auto canvas2 = cv::Mat(meta->frame.rows, meta->frame.cols, meta->frame.type(), cv::Scalar(255, 255, 255));

        for (int i = 0; i < meta->text_targets.size(); i++) {
            auto& text = meta->text_targets[i];

            cv::Point rook_points[4];
            for (int m = 0; m < text->region_vertexes.size(); m++) {
                rook_points[m] =
                    cv::Point(text->region_vertexes[m].first, text->region_vertexes[m].second);
            }

            const cv::Point *ppt[1] = {rook_points};
            int npt[] = {4};

            cv::polylines(canvas1, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, cv::LINE_AA, 0);
            cv::polylines(canvas2, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 1, cv::LINE_AA, 0);
            
            ft2->putText(canvas2, text->text, rook_points[3], 20, cv::Scalar(255, 0, 0), cv::FILLED, cv::LINE_AA, true);
        }

        // copy back to osd frame
        auto roi1 = meta->osd_frame(cv::Rect(0, 0, meta->frame.cols, meta->frame.rows));
        auto roi2 = meta->osd_frame(cv::Rect(0, meta->frame.rows, meta->frame.cols, meta->frame.rows));

        canvas1.copyTo(roi1);
        canvas2.copyTo(roi2);
        
        return meta;
    }
}