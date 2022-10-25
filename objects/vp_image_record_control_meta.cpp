#include "vp_image_record_control_meta.h"

namespace vp_objects {
        
    vp_image_record_control_meta::vp_image_record_control_meta(int channel_index, std::string image_file_name_without_ext, bool osd): 
                                                                vp_control_meta(vp_control_type::IMAGE_RECORD, channel_index),
                                                                image_file_name_without_ext(image_file_name_without_ext),
                                                                osd(osd) {
    }
    
    vp_image_record_control_meta::~vp_image_record_control_meta() {

    }

    std::shared_ptr<vp_meta> vp_image_record_control_meta::clone() {
        // just call copy constructor and return new pointer
        return std::make_shared<vp_image_record_control_meta>(*this);
    }
}