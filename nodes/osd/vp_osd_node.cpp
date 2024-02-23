

#include <opencv2/imgproc.hpp>
#include "vp_osd_node.h"

namespace vp_nodes {
        
    vp_osd_node::vp_osd_node(std::string node_name, std::string font):
                            vp_node(node_name) {
        if (!font.empty()) {
            ft2 = cv::freetype::createFreeType2();
            ft2->loadFontData(font, 0);   
        }       
        this->initialized();
    }
    
    vp_osd_node::~vp_osd_node() {
        deinitialized();
    }
    
    std::shared_ptr<vp_objects::vp_meta> vp_osd_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
        return meta;
    }

    // display logic
    std::shared_ptr<vp_objects::vp_meta> vp_osd_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        // operations on osd_frame
        if (meta->osd_frame.empty()) {
            meta->osd_frame = meta->frame.clone();
        }

        auto& canvas = meta->osd_frame;
        // scan targets
        for (auto& i : meta->targets) {
            // track_id
            auto id = std::to_string(i->track_id);
            auto labels_to_display = i->primary_label;

            // tracked
            if (i->track_id != -1) {
                labels_to_display = "#" + id + " " + labels_to_display;
            }
            
            for (auto& label : i->secondary_labels) {
                labels_to_display += "|" + label;
            }
            
            // draw tracks if size>=2
            if (i->tracks.size() >= 2) {
                for (int n = 0; n < (i->tracks.size() - 1); n++) {
                    auto p1 = i->tracks[n].track_point();
                    auto p2 = i->tracks[n + 1].track_point();
                    cv::line(canvas, cv::Point(p1.x, p1.y), cv::Point(p2.x, p2.y), cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
                }
            }

            cv::rectangle(canvas, cv::Rect(i->x, i->y, i->width, i->height), cv::Scalar(255, 255, 0), 2);
            if (ft2 != nullptr) {
                ft2->putText(canvas, labels_to_display, cv::Point(i->x, i->y), 20, cv::Scalar(255, 0, 255), cv::FILLED, cv::LINE_AA, true);
            }
            else {               
                //cv::putText(canvas, labels_to_display, cv::Point(i->x, i->y), 1, 1, cv::Scalar(255, 0, 255));
                int baseline = 0;
                auto size = cv::getTextSize(labels_to_display, 1, 1.5, 1, &baseline);
                vp_utils::put_text_at_center_of_rect(canvas, labels_to_display, cv::Rect(i->x, i->y - size.height, size.width, size.height), true, 1, 1, cv::Scalar(), cv::Scalar(179, 52, 255), cv::Scalar(179, 52, 255));
            }

            // scan sub targets
            for (auto& sub_target: i->sub_targets) {
                cv::rectangle(canvas, cv::Rect(sub_target->x, sub_target->y, sub_target->width, sub_target->height), cv::Scalar(255));
                if (ft2 != nullptr) {
                    ft2->putText(canvas, sub_target->label, cv::Point(sub_target->x, sub_target->y), 20, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA, true);
                }
                else {
                    cv::putText(canvas, sub_target->label, cv::Point(sub_target->x, sub_target->y), 1, 1, cv::Scalar(0, 0, 255));
                }
            }
            
        }
        return meta;
    }
}