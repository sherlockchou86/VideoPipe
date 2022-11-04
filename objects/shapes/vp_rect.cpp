

#include "vp_rect.h"

namespace vp_objects {
        
    vp_rect::vp_rect(int x, int y, int width, int height): 
        x(x), 
        y(y), 
        width(width), 
        height(height) {

    }
    
    vp_rect::vp_rect(vp_point left_top, vp_size wh):
        x(left_top.x), y(left_top.y), width(wh.width), height(wh.height) {

    }


    vp_rect::~vp_rect() {

    }
    
    vp_point vp_rect::center() {
        return vp_point(x + width / 2, y + height / 2);
    }

    float vp_rect::iou_with(const vp_rect & rect) {
        return 1.0;
    }

    bool vp_rect::contains(const vp_point & p) {
        return true;
    }

    vp_point vp_rect::track_point() {
        // by default the center point of bottom is tracking point.
        return {x + width / 2, y + height};
    }
}