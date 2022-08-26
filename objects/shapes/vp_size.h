
#pragma once

#include <utility>

namespace vp_objects {
    // size(width and height) in 2-dims coordinate system
    class vp_size {
    private:
        /* data */
    public:
        vp_size(int width = 0, int height = 0);
        ~vp_size();


        int width;
        int height;
    };

}