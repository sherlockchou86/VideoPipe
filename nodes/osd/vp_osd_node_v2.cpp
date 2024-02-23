
#include <opencv2/imgproc.hpp>
#include "vp_osd_node_v2.h"
#include "../../utils/vp_utils.h"


namespace vp_nodes {
    
    vp_osd_node_v2::vp_osd_node_v2(std::string node_name, std::string font):vp_node(node_name) {
        if (!font.empty()) {
            ft2 = cv::freetype::createFreeType2();
            ft2->loadFontData(font, 0);   
        }     
        this->initialized();  
    }
    
    vp_osd_node_v2::~vp_osd_node_v2() {
        deinitialized();
    }

    std::shared_ptr<vp_objects::vp_meta> vp_osd_node_v2::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        // operations on osd_frame
        if (meta->osd_frame.empty()) {
            // add a gap at the bottom of osd frame
            meta->osd_frame = cv::Mat(meta->frame.rows + gap_height + padding * 2, meta->frame.cols, meta->frame.type(), cv::Scalar(128, 128, 128));
            
            // initialize by copying frame to osd frame
            auto roi = meta->osd_frame(cv::Rect(0, 0, meta->frame.cols, meta->frame.rows));
            meta->frame.copyTo(roi);
        }
        auto& canvas = meta->osd_frame;

        auto base_left = padding;
        // scan targets
        for (int i = 0; i < meta->targets.size(); i++) {
            auto& target = meta->targets[i];

            // scan sub targets
            for (int j = 0; j < target->sub_targets.size(); j++) {
                auto& sub_target = target->sub_targets[j];
                cv::rectangle(canvas, cv::Rect(sub_target->x, sub_target->y, sub_target->width, sub_target->height), cv::Scalar(255, 255, 255), 2);

                auto roi = canvas(cv::Rect(base_left, meta->osd_frame.rows - padding - gap_height, gap_height, gap_height));
                // white background
                roi = cv::Scalar(255, 255, 255);

                auto ori = canvas(cv::Rect(sub_target->x, sub_target->y, sub_target->width, sub_target->height));                
                cv::Mat tmp = ori.clone();
                int offset_w = 0, offset_h = 0;
                if (tmp.rows > tmp.cols) {
                    cv::resize(tmp, tmp, cv::Size(int(float(gap_height) / tmp.rows * tmp.cols) , gap_height));
                    offset_w = (gap_height - tmp.cols) / 2;
                    offset_h = 0;
                }
                else {
                    cv::resize(tmp, tmp, cv::Size(gap_height, int(float(gap_height) / tmp.cols * tmp.rows)));
                    offset_h = (gap_height - tmp.rows) / 2;
                    offset_w = 0;
                }

                // copy sub target to the bottom of screen
                roi = canvas(cv::Rect(base_left + offset_w, meta->osd_frame.rows - padding - gap_height + offset_h, tmp.cols, tmp.rows));
                tmp.copyTo(roi);

                // line from target to sub target
                cv::line(canvas, cv::Point(target->x + target->width / 2, target->y + target->height), cv::Point(base_left + gap_height / 2, meta->osd_frame.rows - padding - gap_height), cv::Scalar(255, 0, 0), 3, cv::LINE_AA);

                // label of sub target
                auto sub_label = vp_utils::string_split(sub_target->label, '_');
                
                // !!!
                // for plate sub target (color_text)
                if (sub_label.size() == 2) {
                    assert(ft2 != nullptr);
                    ft2->putText(canvas, sub_label[1], cv::Point(base_left + 10, meta->osd_frame.rows - padding - 5), 36, cv::Scalar(0), cv::FILLED, cv::LINE_AA, true);
                }
                base_left += gap_height + padding;                
            }

            // target
            auto labels_to_display = target->primary_label;
            for (auto& label : target->secondary_labels) {
                labels_to_display += "/" + label;
            }
            
            cv::rectangle(canvas, cv::Rect(target->x, target->y, target->width, target->height), cv::Scalar(255, 255, 0), 3);
            if (ft2 != nullptr) {
                ft2->putText(canvas, labels_to_display, cv::Point(target->x, target->y), 25, cv::Scalar(255, 0, 255), cv::FILLED, cv::LINE_AA, true);
            }
            else {               
                cv::putText(canvas, labels_to_display, cv::Point(target->x, target->y), 1, 1, cv::Scalar(255, 0, 255));
            }
        }
        return meta;
    }
}