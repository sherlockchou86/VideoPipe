#include <opencv2/imgproc.hpp>

#include "vp_plate_osd_node.h"

namespace vp_nodes {
    
    vp_plate_osd_node::vp_plate_osd_node(std::string node_name, std::string font): 
                                vp_node(node_name) {
        if (!font.empty()) {
            ft2 = cv::freetype::createFreeType2();
            ft2->loadFontData(font, 0);   
        }       
        this->initialized();
    }
    
    vp_plate_osd_node::~vp_plate_osd_node() {
        deinitialized();
    }

    std::shared_ptr<vp_objects::vp_meta> vp_plate_osd_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
       // operations on osd_frame
        if (meta->osd_frame.empty()) {
            meta->osd_frame = meta->frame.clone();
        }

        auto& canvas = meta->osd_frame;
        cv::rectangle(canvas, cv::Rect(0, canvas.rows - height_his, canvas.cols, height_his), cv::Scalar(240, 200, 220), cv::FILLED);

        // scan targets
        for (auto& i : meta->targets) {
            // plate color and plate text
            auto color_and_text = vp_utils::string_split(i->primary_label, '_');
            if(color_and_text.size() != 2) {
                continue;
            }
            auto& color = color_and_text[0];
            auto& text = color_and_text[1];

            cv::rectangle(canvas, cv::Rect(i->x, i->y, i->width, i->height), draw_colors.at(color), 2);
            if (ft2 != nullptr) {
                ft2->putText(canvas, text_colors.at(color) + " " + text, cv::Point(i->x, i->y), 20, draw_colors.at(color), cv::FILLED, cv::LINE_AA, true);
            }
            else {               
                // ignore
            }   

            // put it to cache
            auto plate = meta->frame(cv::Rect(i->x, i->y, i->width, i->height));
            cv::Mat resized_plate;
            cv::resize(plate, resized_plate, cv::Size(height_his, int((float(height_his) / plate.cols) * plate.rows)));    
            plates_his.push_back(resized_plate);              
        }

        // remove previous history plates if need, since no space to draw
        auto width_need = plates_his.size() * (height_his + gap_his) + gap_his;
        while(width_need >= canvas.cols) {
            plates_his.erase(plates_his.begin());
            width_need = plates_his.size() * (height_his + gap_his) + gap_his;
        }

        // draw history plates at the bottom of screen
        for (int i = 0; i < plates_his.size(); i++) {
            auto& p = plates_his[i];

            auto roi = canvas(cv::Rect((gap_his + height_his) * i + gap_his, canvas.rows - height_his / 2 - p.rows / 2 , height_his, p.rows));
            plates_his[i].copyTo(roi);
        }
        
        return meta;
    }
}