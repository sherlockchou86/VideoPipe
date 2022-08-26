

#include "vp_point.h"

namespace vp_objects {
    
    vp_point::vp_point(int x, int y): x(x), y(y) {

    }
    
    vp_point::~vp_point() {

    }

    float vp_point::distance_with(const vp_point & p) {
        return std::sqrt(std::pow(x-p.x, 2) + std::pow(y-p.y, 2));
    }
}