
#include "vp_control_meta.h"

namespace vp_objects {
        
    vp_control_meta::vp_control_meta(vp_control_type control_type, int channel_index): 
        vp_meta(vp_meta_type::CONTROL, channel_index), control_type(control_type) {
    }
    
    vp_control_meta::~vp_control_meta() {

    }

    std::shared_ptr<vp_meta> vp_control_meta::clone() {
        // just call copy constructor and return new pointer
        return std::make_shared<vp_control_meta>(*this);
    }
}