
#include "vp_frame_target.h"

namespace vp_objects {
        
    vp_frame_target::vp_frame_target(int x, 
                                    int y, 
                                    int width, 
                                    int height, 
                                    int primary_class_id, 
                                    float primary_score, 
                                    int frame_index, 
                                    int channel_index,
                                    std::string primary_label): 
                                    x(x),
                                    y(y),
                                    width(width),
                                    height(height),
                                    primary_class_id(primary_class_id),
                                    primary_score(primary_score),
                                    primary_label(primary_label),
                                    frame_index(frame_index),
                                    channel_index(channel_index),
                                    ba_flags(0) {
    }
    vp_frame_target::vp_frame_target(vp_rect rect,
                                    int primary_class_id, 
                                    float primary_score, 
                                    int frame_index, 
                                    int channel_index,
                                    std::string primary_label):
                                    vp_frame_target(rect.x, 
                                                    rect.y, 
                                                    rect.width, 
                                                    rect.height, 
                                                    primary_class_id, 
                                                    primary_score, 
                                                    frame_index, 
                                                    channel_index, 
                                                    primary_label) {
    
    }


    vp_frame_target::~vp_frame_target() {

    }
    
    std::shared_ptr<vp_frame_target> vp_frame_target::clone() {
        return std::make_shared<vp_frame_target>(*this);
    }

    vp_rect vp_frame_target::get_rect() const{
        return vp_rect(x, y, width, height);
    }
}