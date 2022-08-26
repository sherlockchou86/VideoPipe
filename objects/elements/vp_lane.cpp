

#include "vp_lane.h"

namespace vp_objects {
        
    vp_lane::vp_lane(int element_id, 
                    std::vector<vp_point> vertexs, 
                    vp_lane_type lane_type,
                    std::string element_name, 
                    int ba_abilities_mask):
                    vp_region(element_id, vertexs, element_name, ba_abilities_mask),
                    lane_type(lane_type) {

    }
    
    vp_lane::~vp_lane() {

    }
}