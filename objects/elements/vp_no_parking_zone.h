#pragma once

#include "vp_region.h"

namespace vp_objects {
    // built-in element, a region that no vehicles are allowed to park.
    // this is a built-in element just for easy to use. you can get the same effect using vp_region with explicitly setting ba_abilities_mask = vp_ba::vp_ba_flag::STOP. 
    class vp_no_parking_zone: public vp_region {
    private:
        /* data */
    public:
        vp_no_parking_zone(int element_id, 
                            std::vector<vp_point> vertexs, 
                            std::string element_name = "");
        ~vp_no_parking_zone();
    };
}