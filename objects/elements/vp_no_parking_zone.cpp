
#include "vp_no_parking_zone.h"

namespace vp_objects {
    
    // vp_ba::vp_ba_flag::STOP only
    vp_no_parking_zone::vp_no_parking_zone(int element_id, 
                                            std::vector<vp_point> vertexs, 
                                            std::string element_name):
                                            vp_region(element_id, vertexs, element_name, vp_ba::vp_ba_flag::STOP) {

    }
    
    vp_no_parking_zone::~vp_no_parking_zone() {

    }
    
}