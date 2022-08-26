
#include "vp_region.h"

namespace vp_objects {

    // do not use default ba_abilities_mask=0 here, define which specific ba you want to care for. 
    // for example, setting ba_abilities_mask = vp_ba::vp_ba_flag::STOP | vp_ba::vp_ba_flag::ENTER means it cares for STOP and ENTER.
    vp_region::vp_region(int element_id, 
                        std::vector<vp_point> vertexs, 
                        std::string element_name, 
                        int ba_abilities_mask):
                        vp_frame_element(element_id, element_name, ba_abilities_mask),
                        polygon(vertexs) {
        
    }
    
    vp_region::~vp_region() {
    }

    std::vector<vp_objects::vp_point> vp_region::key_points() {
        return polygon.vertexs;
    }

    std::shared_ptr<vp_frame_element> vp_region::clone() {
        return std::make_shared<vp_region>(*this);
    }

    bool vp_region::contains(const vp_point & p) {
        return this->polygon.contains(p);
    }
}