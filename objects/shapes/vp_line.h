

#pragma once

#include "vp_point.h"

namespace vp_objects {
    // line in 2-dims coordinate system
    class vp_line {
    private:
        /* data */
    public:
        vp_line() = default;
        vp_line(vp_point start, vp_point end);
        ~vp_line();

        vp_point start;
        vp_point end;

        // distance between start and end point
        float length();
    };

}