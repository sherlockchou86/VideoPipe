
#pragma once

#include "vp_region.h"

namespace vp_objects {
    // lane type
    enum vp_lane_type {
        DRIVE,       // drive lane
        PASS,        // pass lane
        EMERGENCY    // emergency lane
    };

    // built-in element, lane in road.
    // sample to show how to implement a custom element derived from vp_region.
    class vp_lane: public vp_region {
    private:
        /* data */
    public:
        vp_lane(int element_id, 
                std::vector<vp_point> vertexs, 
                vp_lane_type lane_type = vp_lane_type::DRIVE,
                std::string element_name = "", 
                int ba_abilities_mask = 0);
        ~vp_lane();

        const vp_lane_type lane_type;
    };
}