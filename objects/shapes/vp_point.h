
#pragma once

#include <utility>
#include <cmath>

namespace vp_objects {
    // point in 2-dims coordinate system
    class vp_point
    {
    private:
        /* data */
    public:
        vp_point(int x = 0, int y = 0);
        ~vp_point();

        int x;
        int y;

        // distance between 2 points
        float distance_with(const vp_point & p);
    };    
}