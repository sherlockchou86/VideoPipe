#pragma once

#include <tuple>

#include "vp_point.h"
#include "vp_size.h"

namespace vp_objects {
    // rect in 2-dims coordinate system
    class vp_rect {
    private:
        /* data */
    public:
        vp_rect() = default;
        vp_rect(int x, int y, int width, int height);
        vp_rect(vp_point left_top, vp_size wh);
        ~vp_rect();

        int x;
        int y;
        int width;
        int height;

        // get center point of the rect
        vp_point center();

        // get track point of the rect
        // track point is used to locate the target(represented by the rect)
        vp_point track_point();

        // calculate the iou with another rect
        float iou_with(const vp_rect & rect);

        // check if the rect contains a point
        bool contains(const vp_point & p);
    };

}