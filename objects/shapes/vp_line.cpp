

#include "vp_line.h"

namespace vp_objects {
    vp_line::vp_line(vp_point start, vp_point end): start(start), end(end) {

    }
    
    vp_line::~vp_line() {

    }

    float vp_line::length() {
        return start.distance_with(end);
    }
}