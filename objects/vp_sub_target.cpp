
#include "vp_sub_target.h"

namespace vp_objects {
    
    vp_sub_target::vp_sub_target(int x, 
                        int y, 
                        int width, 
                        int height, 
                        int class_id, 
                        float score, 
                        std::string label, 
                        int frame_index, 
                        int channel_index):
                        x(x),
                        y(y),
                        width(width),
                        height(height),
                        class_id(class_id),
                        score(score),
                        label(label),
                        frame_index(frame_index),
                        channel_index(channel_index) {
    }
    
    vp_sub_target::~vp_sub_target() {
    }    

    std::shared_ptr<vp_sub_target> vp_sub_target::clone() {
        return std::make_shared<vp_sub_target>(*this);
    }

    vp_rect vp_sub_target::get_rect() const {
        return vp_rect(x, y, width, height);
    }
}