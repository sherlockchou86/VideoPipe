#include <iterator>

#include "vp_frame_meta.h"

namespace vp_objects {
        
    vp_frame_meta::vp_frame_meta(cv::Mat frame, int frame_index, int channel_index, int original_width, int original_height, int fps): 
        vp_meta(vp_meta_type::FRAME, channel_index), 
        frame_index(frame_index), 
        original_width(original_width),
        original_height(original_height),
        fps(fps),
        frame(frame) {
            assert(!frame.empty());
    }
    
    // copy constructor of vp_frame_meta would NOT be called at most time.
    // only when it flow through vp_split_node with vp_split_node::split_with_deep_copy==true.
    // in fact, all kinds of meta would NOT be copyed in its lifecycle, we just pass them by poniter most time.
    vp_frame_meta::vp_frame_meta(const vp_frame_meta& meta): 
        vp_meta(meta),
        frame_index(meta.frame_index),
        original_width(meta.original_width),
        original_height(meta.original_height),
        fps(meta.fps) {
            // deep copy frame data
            this->frame = meta.frame.clone();
            this->osd_frame = meta.osd_frame.clone();
            // deep copy elements
            for(auto& i : meta.elements) {
                this->elements.push_back(i->clone());
            }
            // deep copy targets
            for(auto& i: meta.targets) {
                this->targets.push_back(i->clone());
            }
            // deep copy pose targets
            for(auto& i: meta.pose_targets) {
                this->pose_targets.push_back(i->clone());
            }
            // deep copy face targets
            for(auto& i: meta.face_targets) {
                this->face_targets.push_back(i->clone());
            }
            // deep copy text targets
            for(auto& i: meta.text_targets) {
                this->text_targets.push_back(i->clone());
            }
            // deep copy ba_flag_map
            // we need re-find map relationship of pointers between elements and targets since they are deep copyed above.
            for(auto& i: meta.ba_flags_map) {
                auto e = std::find_if(std::begin(this->elements), 
                                        std::end(this->elements), 
                                        [&](std::shared_ptr<vp_objects::vp_frame_element> element) {return element->element_id == std::get<0>(i)->element_id;});
                auto t = std::find_if(std::begin(this->targets), 
                                        std::end(this->targets), 
                                        [&](std::shared_ptr<vp_objects::vp_frame_target> target) {return target->track_id == std::get<1>(i)->track_id;});
                if (e != std::end(this->elements) && t != std::end(this->targets)) {
                    this->ba_flags_map.push_back(std::make_tuple(*e, *t, std::get<2>(i)));
                }
            }
    }
    
    vp_frame_meta::~vp_frame_meta() {

    }

    std::shared_ptr<vp_meta> vp_frame_meta::clone() {
        // just call copy constructor and return new pointer
        return std::make_shared<vp_frame_meta>(*this);
    }
}