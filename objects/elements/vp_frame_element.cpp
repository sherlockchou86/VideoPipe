

#include "vp_frame_element.h"

namespace vp_objects {
        
    vp_frame_element::vp_frame_element(int element_id, 
        std::string element_name, 
        int ba_abilities_mask): 
        element_id(element_id),
        element_name(element_name.empty()?std::to_string(element_id):element_name),
        ba_abilities_mask(ba_abilities_mask) {

    }
    
    vp_frame_element::~vp_frame_element() {
    }
    
    bool vp_frame_element::check_ba_ability(vp_ba::vp_ba_flag flag) {
        return this->ba_abilities_mask & flag == flag;
    }
}